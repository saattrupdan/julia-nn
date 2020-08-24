function mse_loss(preds::Array, target::Array)::AbstractFloat
  sum((preds - target) .^ 2)
end
