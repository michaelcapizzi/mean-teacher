import numpy as np
import math
import itertools
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function

from .utils import export, parameter_count


@export
def cifar_shakeshake26(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='shift_conv', **kwargs)
    return model


@export
def resnext152(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet224x224(BottleneckBlock,
                          layers=[3, 8, 36, 3],
                          channels=32 * 4,
                          groups=32,
                          downsample='basic', **kwargs)
    return model



class ResNet224x224(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 4
        self.downsample_mode = downsample
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, channels * 8, groups, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(block.out_channels(
            channels * 8, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(
            channels * 8, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


class ResNet32x32(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BottleneckBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        if groups > 1:
            return 2 * planes
        else:
            return 4 * planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv_a1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn_a2 = nn.BatchNorm2d(planes)
        self.conv_a3 = nn.Conv2d(planes, self.out_channels(
            planes, groups), kernel_size=1, bias=False)
        self.bn_a3 = nn.BatchNorm2d(self.out_channels(planes, groups))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, residual = x, x

        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = self.relu(a)
        a = self.conv_a2(a)
        a = self.bn_a2(a)
        a = self.relu(a)
        a = self.conv_a3(a)
        a = self.bn_a3(a)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(residual + a)


class ShakeShakeBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + ab


class Shake(Function):
    @classmethod
    def forward(cls, ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
        gate = inp1.new(*gate_size)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1. - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1,
                                                              grad_output.dim() - 1)]
        gate = Variable(grad_output.data.new(*gate_size).uniform_(0, 1))
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1 - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training


def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)


class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=2 * in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2],
                       x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def get_input_size(dict_of_embedding_classes):
    """
    Determine the required input size by concatenating embeddings
    :return: <int>
    """
    total_input_size = 0
    for e, v in dict_of_embedding_classes.items():
        total_input_size += v.embedding_dim
    return total_input_size


@export
class LSTM(nn.Module):
    def __init__(self, num_layers, input_embeddings, hidden_size, output_size,
                 batch_size, dropout_rate=None, word_dropout_rate=None,
                 bi_directional=True, use_gru=True, use_gpu=True):
        """
        :param num_layers: number of layers to the model
        :param input_embeddings: <OrderedDict> of Embedding classes for each input to model
                                    k=embedding_layer_name, v=Embedding class
        :param hidden_size: size of hidden layer and c-state in LSTM
        :param output_size: number of labels
        :param batch_size: size of batch
        :param dropout_rate: dropout rate to apply to each layer
        :param word_dropout_rate: word-level dropout rate to apply to each layer
        :param bi_directional: If True, will be bi-directional
        :param use_gru: If True, will use GRU instead of LSTM
        :param use_gpu: If True, will activate .cuda() for layers
        """
        super().__init__()
        self.num_layers = num_layers
        self.input_size = get_input_size(input_embeddings)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.bi_directional = bi_directional
        self.model = getattr(torch.nn, "LSTM" if not use_gru else "GRU")(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate if dropout_rate != None else 0.0,
            bidirectional=self.bi_directional,

        )
        self.word_level_dropout_rate = word_dropout_rate
        self.word_level_dropout_layers = self._build_word_dropout_layers()
        self.projection_layer_classification = torch.nn.Linear(
            in_features=self.hidden_size if not self.bi_directional else self.hidden_size * 2,
            out_features=output_size
        )
        self.projection_layer_consistency = torch.nn.Linear(
            in_features=self.hidden_size if not self.bi_directional else self.hidden_size * 2,
            out_features=output_size
        )
        self.input_embeddings = self._process_embeddings(input_embeddings)
        if use_gpu:
            self.cuda()
        self.use_gpu = use_gpu

    def _process_embeddings(self, embedding_dict):
        # add modules
        for k,v in embedding_dict.items():
            self.add_module(k, v)
        return embedding_dict

    def _build_word_dropout_layers(self):
        """
        Builds word-level dropout layers to be applied to LSTM
        """
        if self.word_level_dropout_rate:
            word_level_dropout_layers = OrderedDict()
            for i in range(self.num_layers):
                word_level_dropout_layers[i] = torch.nn.Dropout2d(self.word_level_dropout_rate)
            return word_level_dropout_layers

    def init_hidden(self):
        """
        Reset hidden (h_*) and c-state (c_*)
        with shape of(num_layers, minibatch_size, hidden_dim)
        """
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),  # h_*
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size)   # c_*
            )

    def forward(self, xs):
        """
        Runs a single pass through the network
        :param xs: <OrderedDict> of inputs corresponding to each embedding layer in self.input_embeddings
                    key=name_of_input_embedding, value=input
        :return: <FloatTensor>
        """
        inputs = OrderedDict()
        # run through embedding layers
        for xk, xv in xs.items():
            inputs[xk] = self.input_embeddings[xk](xv)
        # concatenate
        input_ = torch.cat(list(inputs.values()), -1)
        # apply word-level dropout
        # TODO implement for all layers
        if self.word_level_dropout_layers:
            input_ = self.word_level_dropout_layers[0](input_)   # currently only applying at input layer
        # run through LSTM
        lstm_out, _ = self.model(input_)
        # apply projection
        final_out_classification = self.projection_layer_classification(lstm_out)
        final_out_consistency = self.projection_layer_consistency(lstm_out)
        return final_out_classification[:, -1], final_out_consistency[:, -1]


@export
class DAN(nn.Module):
    def __init__(self, num_layers, input_embedding_bags, hidden_size, output_size,
                 batch_size, activation="ReLU", dropout_rate=None, word_dropout_rate=None, use_gpu=True):
        """
        Architecture described in this paper: http://www.cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf
        :param num_layers: number of hidden layers in the model
        :param input_embedding_bags: <OrderedDict> of EmbeddingBag classes for each input to model
                                    k=embedding_layer_name, v=EmbeddingBag class
        :param hidden_size: size of hidden layer
        :param output_size: number of labels
        :param batch_size: size of batch
        :param activation: type of activation layer
        :param dropout_rate: dropout rate to apply to each layer
        :param word_dropout_rate: word-level dropout rate to apply to input layer
        :param use_gpu: If True, will activate .cuda() for layers
        """
        super().__init__()
        if num_layers < 2:
            raise Exception("number of layers for DAN architecture must be > 2")
        self.num_layers = num_layers
        self.input_size = get_input_size(input_embedding_bags)
        self.input_embedding_bags = input_embedding_bags
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_layers = self._build_dropout_layers(num_layers, dropout_rate)
        self.word_level_dropout_rate = word_dropout_rate
        self.hidden_layers = self._build_hidden_layers(num_layers, self.input_size, hidden_size, output_size)
        self.activation_layers = self._build_activation_layers(num_layers, activation)
        self._process_parameters()
        if use_gpu:
            self.cuda()
        self.use_gpu = use_gpu

    @staticmethod
    def _build_dropout_layers(num_layers, d_rate):
        if not d_rate:
            d_rate = 0.0
        dropout_layers = OrderedDict()
        for i in range(num_layers):
            dropout_layers[i] = torch.nn.Dropout(d_rate)
        return dropout_layers

    @staticmethod
    def _build_hidden_layers(num_layers, input_size, hidden_size, output_size):
        hidden_layers = OrderedDict()
        for i in range(num_layers):
            if i == 0:
                hidden_layers[0] = torch.nn.Linear(
                    in_features=input_size,
                    out_features=hidden_size
                )
            elif i != num_layers - 1:
                hidden_layers[i] = torch.nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size
                )
            else:
                # for last layer make *two* matrices for *two* outputs
                hidden_layers[i] = {}
                hidden_layers[i]["classification"] = torch.nn.Linear(
                    in_features=hidden_size,
                    out_features=output_size
                )
                hidden_layers[i]["consistency"] = torch.nn.Linear(
                    in_features=hidden_size,
                    out_features=output_size
                )
        return hidden_layers

    @staticmethod
    def _build_activation_layers(num_layers, activation_type):
        activation = getattr(nn, activation_type)
        activation_layers = OrderedDict()
        for i in range(num_layers):
            activation_layers[i] = activation()
        return activation_layers

    def _process_parameters(self):
        # add embeddings
        for k,v in self.input_embedding_bags.items():
            self.add_module(k, v)
        # add hidden layers
        for k,v in self.hidden_layers.items():
            if not isinstance(v, dict):
                self.add_module("hidden_{}".format(k), v)
            else:
                self.add_module("hidden_{}_classification".format(k), v["classification"])
                self.add_module("hidden_{}_consistency".format(k), v["consistency"])

    def _apply_word_level_dropout(self, xs):
        """
        Randomly drops indices with a probability of self.word_dropout_rate
        :param xs: <OrderedDict> of inputs corresponding to each embedding layer in self.input_embedding_bags
                    key=name_of_input_embedding, value=input
        :return: <LongTensor>
        """
        TENSOR_MODULE = torch if not self.use_gpu else torch.cuda
        if self.word_level_dropout_rate:
            dropped_out_values = OrderedDict()
            for n, v in xs.items():
                shape_ = v.shape[1]
                # determine which indexes to keep
                dropout_tensor = getattr(TENSOR_MODULE, "LongTensor")(
                    np.full((1, shape_), 1 - self.word_level_dropout_rate)
                )
                dropout_tensor.bernoulli_().type(torch.LongTensor)
                nonzero_values = dropout_tensor.nonzero()[:,1]
                dropped_out_values[n] = v[:,:][:,nonzero_values]
            return dropped_out_values
        else:
            return xs

    def forward(self, xs):
        """
        Runs a single pass through the network
        :param xs: <OrderedDict> of inputs corresponding to each embedding layer in self.input_embedding_bags
                    key=name_of_input_embedding, value=input
        :return: <FloatTensor>
        """
        inputs = OrderedDict()
        # apply word_level_dropout
        dropped_xs = self._apply_word_level_dropout(xs)
        # run through embedding layers
        for xk, xv in dropped_xs.items():
            inputs[xk] = self.input_embedding_bags[xk](xv)
        # concatenate
        input_ = torch.cat(list(inputs.values()), -1)
        # run through hidden layers and dropout layers
        hidden_ = input_
        for i in range(self.num_layers):
            h = self.hidden_layers[i]
            d = self.dropout_layers[i]
            a = self.activation_layers[i]
            if i != self.num_layers - 1:
                hidden_ = d(hidden_)
                hidden_ = h(hidden_)
                hidden_ = a(hidden_)
            else:
                hidden_ = d(hidden_)
                out_classification = h["classification"](hidden_)
                out_consistency = h["consistency"](hidden_)
        return out_classification, out_consistency
