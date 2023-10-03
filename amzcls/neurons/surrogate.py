from spikingjelly.activation_based.surrogate import (
    PiecewiseQuadratic, PiecewiseExp, Sigmoid, SoftSign, ATan, NonzeroSignLogAbs,
    Erf, PiecewiseLeakyReLU, SquarewaveFourierSeries, S2NN, QPseudoSpike, LeakyKReLU,
    FakeNumericalGradient, LogTailedReLU
)

from .builder import SURROGATE

for surrogate in (
        PiecewiseQuadratic, PiecewiseExp, Sigmoid, SoftSign, ATan, NonzeroSignLogAbs,
        Erf, PiecewiseLeakyReLU, SquarewaveFourierSeries, S2NN, QPseudoSpike, LeakyKReLU,
        FakeNumericalGradient, LogTailedReLU
):
    SURROGATE.register_module(module=surrogate)
