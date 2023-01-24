from mmcls.models.builder import ATTENTION as MMCLS_ATTENTION
from mmcls.models.builder import MODELS as MMCLS_MODELS
from mmcv.utils import Registry

from ..version import USE_MMCLS

if USE_MMCLS:
    MODELS = MMCLS_MODELS
    ATTENTION = MMCLS_ATTENTION
else:
    MODELS = Registry('models', parent=MMCLS_MODELS)
    ATTENTION = Registry('attention', parent=MMCLS_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)
