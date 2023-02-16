import torch
from spikingjelly.activation_based import layer, neuron, functional
from torch import nn


class SpikeTRF(nn.Module):
    def __init__(self, in_dim, qk_dim, v_dim):
        super(SpikeTRF, self).__init__()
        self.layer_q = layer.Linear(in_dim, qk_dim)
        self.neuron_q = neuron.LIFNode()
        self.layer_k = layer.Linear(in_dim, qk_dim)
        self.layer_v = layer.Linear(in_dim, v_dim)
        self.neuron_v = neuron.LIFNode()
        self.neuron_o = neuron.LIFNode()

    def forward(self, x):
        q = self.neuron_q(self.layer_q(x))
        k = self.layer_k(x)
        v = self.neuron_v(self.layer_v(x))
        attention = torch.einsum('...ij,...kj->...ik', q, k)
        result = self.neuron_o(attention @ v)
        return result


if __name__ == '__main__':
    # [B, N, L]
    inputs = torch.randn((1, 32, 16))
    trf = SpikeTRF(16, 16, 64)
    out = trf(inputs)
    print(out.shape)
    # [T, B, N, L]
    inputs = torch.randn((4, 1, 32, 16))
    functional.set_step_mode(trf, step_mode='m')
    out = trf(inputs)
    print(out.shape)