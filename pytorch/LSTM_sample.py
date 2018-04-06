from mean_teacher import architectures
from torch import FloatTensor, LongTensor
import torch.autograd
import torch.nn

# build embedding layer
print("building embedding layer")
embedding_layer = torch.nn.Embedding(10, 7)

# build LSTM
print("building LSTM")
LSTM = architectures.LSTM(
    num_layers=1,
    input_embeddings={"input": embedding_layer},
    hidden_size=3,
    output_size=2,
    batch_size=1,
    use_gpu=False
)
# sample input
# 1 x 5 x 1
# batch x seq_length x input_dim
sample_in = LongTensor([
    [0,1,2,3,4]
])
sample_in = torch.autograd.Variable(sample_in, requires_grad=False)

print(LSTM.forward({"input": sample_in}))
