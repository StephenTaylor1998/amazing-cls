_base_ = [
    './surrogates.py'
]

# inherit from ./surrogates.py
Sigmoid = dict()

neuron_cfg = dict(
    type='IzhikevichNode',
    tau=2.,
    v_c=0.8,
    a0=1.,
    v_threshold=1.,
    v_reset=0.,
    v_rest=-0.1,
    w_rest=0.,
    tau_w=2.,
    a=0.,
    b=0.,
    surrogate_function=Sigmoid,
    detach_reset=False,
    step_mode='s',
    backend='torch',
    store_v_seq=False
)
