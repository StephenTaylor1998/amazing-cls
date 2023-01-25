_base_ = [
    './surrogates.py'
]

# inherit from ./surrogates.py
Sigmoid = dict()

neuron_cfg = dict(
    type='IFNode',
    v_threshold = 1.,
    v_reset = 0.,
    surrogate_function=Sigmoid,
    detach_reset= False,
    step_mode = 's',
    backend = 'torch',
    store_v_seq = False
)
