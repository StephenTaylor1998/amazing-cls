PiecewiseQuadratic = dict(
    alpha=1.0,
    spiking=True
)
PiecewiseExp = dict(
    alpha=1.0,
    spiking=True
)
Sigmoid = dict(
    alpha=4.0,
    spiking=True
)
SoftSign = dict(
    alpha=2.0,
    spiking=True
)
ATan = dict(
    alpha=2.0,
    spiking=True
)
NonzeroSignLogAbs = dict(
    alpha=1.0,
    spiking=True
)
Erf = dict(
    alpha=2.0,
    spiking=True
)
PiecewiseLeakyReLU = dict(
    w=1.,
    c=0.01,
    spiking=True
)
SquarewaveFourierSeries = dict(
    n=2,
    T_period=8,
    spiking=True
)
S2NN = dict(
    alpha=4.,
    beta=1.,
    spiking=True
)
QPseudoSpike = dict(
    alpha=2.0,
    spiking=True
)
LeakyKReLU = dict(
    spiking=True,
    leak=0.,
    k=1.
)
FakeNumericalGradient = dict(
    alpha=0.3
)
LogTailedReLU = dict(
    alpha=0.,
    spiking=True
)
