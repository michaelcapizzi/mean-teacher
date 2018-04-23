from mean_teacher import architectures
from torch import FloatTensor, LongTensor
import torch.autograd
import torch.nn

USE_GPU = False

# build embedding_bag layer
print("building embedding_bag layer")
embedding_layer_1 = torch.nn.EmbeddingBag(10, 7)
embedding_layer_2 = torch.nn.EmbeddingBag(4, 2)

# build DAN
print("building DAN")
DAN = architectures.DAN(
    num_layers=8,
    input_embedding_bags={"input_1": embedding_layer_1, "input_2": embedding_layer_2},
    hidden_size=3,
    output_size=2,
    batch_size=2,
    dropout_rate=0.4,
    word_dropout_rate=0.4,
    use_gpu=False
)
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

out_cl, out_co = DAN.forward({"input_1": sample_in_1, "input_2": sample_in_2})
print("final outs")
print(out_cl, out_co)
