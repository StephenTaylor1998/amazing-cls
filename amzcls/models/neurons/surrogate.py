from spikingjelly.activation_based.surrogate import (
    PiecewiseQuadratic, PiecewiseExp, Sigmoid, SoftSign, ATan, NonzeroSignLogAbs,
    Erf, PiecewiseLeakyReLU, SquarewaveFourierSeries, S2NN, QPseudoSpike, LeakyKReLU,
    FakeNumericalGradient, LogTailedReLU
)

from .builder import SURROGATE

# PiecewiseQuadratic = SURROGATE.register_module(PiecewiseQuadratic)
# PiecewiseExp = SURROGATE.register_module(PiecewiseExp)
# Sigmoid = SURROGATE.register_module(Sigmoid)
# SoftSign = SURROGATE.register_module(SoftSign)
# ATan = SURROGATE.register_module(ATan)
# NonzeroSignLogAbs = SURROGATE.register_module(NonzeroSignLogAbs)
# Erf = SURROGATE.register_module(Erf)
# PiecewiseLeakyReLU = SURROGATE.register_module(PiecewiseLeakyReLU)
# SquarewaveFourierSeries = SURROGATE.register_module(SquarewaveFourierSeries)
# S2NN = SURROGATE.register_module(S2NN)
# QPseudoSpike = SURROGATE.register_module(QPseudoSpike)
# LeakyKReLU = SURROGATE.register_module(LeakyKReLU)
# FakeNumericalGradient = SURROGATE.register_module(FakeNumericalGradient)
# LogTailedReLU = SURROGATE.register_module(LogTailedReLU)

for surrogate in (
    PiecewiseQuadratic, PiecewiseExp, Sigmoid, SoftSign, ATan, NonzeroSignLogAbs,
    Erf, PiecewiseLeakyReLU, SquarewaveFourierSeries, S2NN, QPseudoSpike, LeakyKReLU,
    FakeNumericalGradient, LogTailedReLU
):
    SURROGATE.register_module(surrogate)
