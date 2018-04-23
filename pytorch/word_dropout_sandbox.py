from mean_teacher import architectures
from torch import FloatTensor, LongTensor
import torch.autograd
import torch.nn
import numpy as np

# build embedding_bag layer
print("building embedding_bag layer")
embedding_layer = torch.nn.EmbeddingBag(10, 7)

# sample input
# 1 x 5 x 1
# batch x seq_length x input_dim
sample_in = LongTensor([
    [0,1,2,3,4],
    [4,3,2,1,0]
])

print("sample_in", sample_in)
shape_ = sample_in.shape[1]
dropout_tensor = torch.FloatTensor(np.full((1, shape_), 0.50))
print("probs", dropout_tensor)
dropout_tensor.bernoulli_().type(torch.LongTensor)
print("bernoulli", dropout_tensor)
nonzero = dropout_tensor.nonzero()[:,1]
print("nonzero", nonzero)
print("sample + nonzero", sample_in[:,:][:,nonzero])
