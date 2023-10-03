from copy import deepcopy

from ..registry import MODELS

NODES = MODELS
SURROGATE = MODELS


def build_node(cfg):
    _cfg = deepcopy(cfg)
    if 'surrogate_function' in _cfg.keys():
        surrogate = build_surrogate(_cfg['surrogate_function'])
        _cfg['surrogate_function'] = surrogate
    return NODES.build(_cfg)


def build_surrogate(cfg):
    return SURROGATE.build(cfg)
