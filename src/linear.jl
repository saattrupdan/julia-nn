struct Linear
  W
  b
  σ
end

function (m::Linear)(x::Array)::Array
  m.σ(m.W * x + m.b)
end

# Convenience constructor
function Linear(in::Integer, out::Integer)::Linear
  Linear(randn(out, in), randn(out), x -> x)
end

# Convenience constructor if an activation function is specified
function Linear(in::Integer, out::Integer, σ::Function)::Linear
  Linear(randn(out, in), randn(out), σ)
end
