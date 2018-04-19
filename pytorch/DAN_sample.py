from mean_teacher import architectures
from torch import FloatTensor, LongTensor
import torch.autograd
import torch.nn

USE_GPU = False

# build embedding_bag layer
print("building embedding_bag layer")
embedding_layer = torch.nn.EmbeddingBag(10, 7)

# build DAN
print("building DAN")
DAN = architectures.DAN(
    num_layers=8,
    input_embedding_bags={"input": embedding_layer},
    hidden_size=3,
    output_size=2,
    batch_size=1,
    dropout_rate=0.4,
    word_dropout_rate=0.4,
    use_gpu=False
)
# sample input
# 1 x 5 x 1
# batch x seq_length x input_dim
sample_in = LongTensor([
    [0,1,2,3,4]
])
if USE_GPU:
    sample_in.cuda()

sample_in = torch.autograd.Variable(sample_in, requires_grad=False)

out_ = DAN.forward({"input": sample_in})
print(out_.shape)
