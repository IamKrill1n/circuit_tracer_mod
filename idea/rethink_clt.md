the residual⡀vector is decomposed⠈into sparse features in a⠈CLT, but a vector
  later in the context window, should accumulate information from previous     ⠄
  context positions, so the vector should⠈have more meaning. I don't know if the
 ⠈sparsity loss used during training normalize to the token position or not, but
  at least when we prune the attribution graph by influence, it always has this
  shape [Image #1] where later⠈token positions have more influential features (at
 ⢀least in later layers). I still 