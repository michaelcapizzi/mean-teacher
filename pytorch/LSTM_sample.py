from mean_teacher import architectures
from torch import FloatTensor, LongTensor
import torch.autograd
import torch.nn

USE_GPU = False

# build embedding layer
print("building embedding layer")
embedding_layer_1 = torch.nn.Embedding(10, 7)
embedding_layer_2 = torch.nn.Embedding(4, 2)

# build LSTM
print("building LSTM")
LSTM = architectures.LSTM(
    num_layers=1,
    input_embeddings={"input_1": embedding_layer_1, "input_2": embedding_layer_2},
    hidden_size=3,
    output_size=2,
    batch_size=1,
    use_gpu=USE_GPU
)

for n,p in LSTM.named_parameters():
    print(n, p)

# sample input
# 1 x 5 x 1
# batch x seq_length x input_dim
sample_in_1 = LongTensor([
    [0,1,2,3,9],
    [4,3,2,1,6]
])
sample_in_2 = LongTensor([
    [0,1,2,3,3],
    [1,3,2,1,1]
])

if USE_GPU:
    sample_in_1.cuda()
    sample_in_2.cuda()

sample_in_1 = torch.autograd.Variable(sample_in_1, requires_grad=False)
sample_in_2 = torch.autograd.Variable(sample_in_2, requires_grad=False)

out_, _= LSTM.forward({"input_1": sample_in_1, "input_2": sample_in_2})

print(out_.shape, out_[:, -1].shape)
