using LightGraphs
using Flux

function generate_training_samples(graph::AbstractGraph, start_node::Integer,
                                   walk_len::Integer, window::Integer)::Tuple
  # Generate random walk
  walk = randomwalk(graph, start_node, walk_len)

  # Collect nodes and their neighbours within the random walk
  nodes = Vector{Integer}()
  neighbours = Vector{Integer}()
  for (idx, node) in enumerate(walk)
    window_left = max(1, idx - window)
    window_right = min(size(walk, 1), idx + window)
    for neighbour in walk[window_left:window_right]
      if neighbour != node
        push!(nodes, node)
        push!(neighbours, neighbour)
      end
    end
  end
  nodes, neighbours
end

function train_model(graph::AbstractGraph; walk_len::Integer=10,
                     window::Integer=3, num_epochs::Integer=1, 
                     emb_dim::Integer=10)

  # Collect all the nodes in the graph
  num_nodes = nv(graph)
  all_nodes = collect(vertices(graph))

  # Build the model 
  model = Chain(Dense(num_nodes, emb_dim), Dense(emb_dim, num_nodes))

  # Define our loss function, being cross entropy on a one-hot encoding
  # of the nodes
  function criterion(nodes::Array{Integer}, 
                     neighbours::Array{Integer})::AbstractFloat
    nodes = Flux.onehotbatch(nodes, all_nodes)
    neighbours = Flux.onehotbatch(neighbours, all_nodes)
    Flux.Losses.logitcrossentropy(model(nodes), neighbours)
  end

  # Set up the optimiser and fetch the parameters of the model
  optimiser = Flux.Optimise.ADAMW(3e-4)
  params = Flux.params(model)
 
  # Main training loop
  losses = []
  for epoch in 1:num_epochs
    avg_loss = 0

    # Looping over all nodes in the graph
    for start_node in vertices(graph)

      # Generate samples using the helper function
      (nodes, neighbours) = generate_training_samples(graph, start_node, 
                                                      walk_len, window)

      # Enable gradient computation and compute the loss
      gradients = gradient(params) do
        loss = criterion(nodes, neighbours)
        avg_loss += loss
        return loss
      end

      # Backpropagate the loss through the network
      Flux.Optimise.update!(optimiser, params, gradients)
    end

    # Get the average loss of the epoch, save it to `losses`
    # and print out the status
    avg_loss /= nv(graph)
    append!(losses, avg_loss)
    println("Epoch ", epoch, " - Average loss: ", avg_loss)
  end

  # Return the model along with the array of all the average losses
  model, losses
end

function embed(graph::AbstractGraph, model)::Array
  all_nodes = collect(vertices(graph))
  nodes = Flux.onehotbatch(all_nodes, all_nodes)
  model[1](nodes)
end
