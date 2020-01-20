import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, conditional_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.conditional_shape = conditional_shape
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
            self.weight_layer = nn.Linear(conditional_shape, normalized_shape[-1], bias=False)
            self.bias_layer = nn.Linear(conditional_shape, normalized_shape[-1], bias=False)
            self.hidden_dense = nn.Linear(conditional_shape, conditional_shape)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.zeros_(self.weight_layer.weight)
            nn.init.zeros_(self.bias_layer.weight)
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)


    def forward(self, inputs):
        x, cond = inputs
        cond = self.hidden_dense(cond)
        for _ in range(len(x.shape) - len(cond.shape)):
            cond.unsqueeze_(1)
        weight = self.weight_layer(cond) + self.weight
        bias = self.bias_layer(cond) + self.bias

        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.mean((x-mean)**2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        outputs = (x - mean) / std
        outputs = outputs * weight + bias
        return outputs

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class ContentExtraction(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(ContentExtraction, self).__init__()
        self.input_size = embed_size
        self.outptu_size = hidden_size
        self.LSTM = nn.LSTM(embed_size, hidden_size, batch_first=True)


    def forward(self, content_embed):
        _, (h_n, c_n) = self.LSTM(content_embed)
        h_n = h_n.transpose(1, 0)
        return h_n
