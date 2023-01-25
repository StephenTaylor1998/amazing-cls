_base_ = [
    './surrogates.py'
]

# inherit from ./surrogates.py
Sigmoid = dict()

neuron_cfg = dict(
    type='EIFNode',
    tau = 2.,
    delta_T = 1.,
    theta_rh = .8,
    v_threshold = 1.,
    v_rest = 0.,
    v_reset = -0.1,
    surrogate_function = Sigmoid,
    detach_reset = False,
    step_mode = 's'
)
