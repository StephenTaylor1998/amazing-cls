import torch
from spikingjelly.activation_based import layer, neuron, functional
from torch import nn


class SpikeTRF(nn.Module):
    def __init__(self, in_dim, qk_dim, v_dim):
        super(SpikeTRF, self).__init__()
        self.layer_q = layer.Linear(in_dim, qk_dim)
        self.neuron_q = neuron.IFNode()
        self.layer_k = layer.Linear(in_dim, qk_dim)
        self.layer_v = layer.Linear(in_dim, v_dim)
        self.neuron_v = neuron.IFNode()
        self.neuron_o = neuron.IFNode()

    def forward(self, x):
        q = self.neuron_q(self.layer_q(x))
        k = self.layer_k(x)
        v = self.neuron_v(self.layer_v(x))

        print(v[-1, -1])
        attention = torch.einsum('ibj,kbj->ibk', q, k)
        print(attention[0, 0])
        result = self.neuron_o(torch.einsum('ibj,jbk->ibk', attention, v))
        return result


if __name__ == '__main__':
    trf = SpikeTRF(64, 64, 16)
    # [T, B, N]
    inputs = torch.randn((10, 1, 64))
    out = trf(inputs)
    print(out[-1, -1])

