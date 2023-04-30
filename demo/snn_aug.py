import random

import torch

from amzcls.datasets import PIPELINES

if __name__ == '__main__':
    # piplines = PIPELINES.build(dict(type='SNNAugment', keys=['img']))
    # print(piplines)
    # inputs = dict(
    #     img=torch.randn((4, 2, 128, 128))
    # )
    # out = piplines(inputs)
    # print(out)

    # piplines = PIPELINES.build(
    #     dict(type='TimeSample', keys=['img'], time_step=16, sample_step=12, use_rand=True)
    # )
    # print(piplines)
    # inputs = dict(
    #     img=torch.randn((16, 2, 128, 128))
    # )
    # out = piplines(inputs)
    # print(out['img'].shape)
    #
    # for i in range(100):
    #     inputs = dict(
    #         img=torch.randn((16, 2, 128, 128))
    #     )
    #     out = piplines(inputs)
    #     print(out['img'].shape)

    num_bins = 31
    augmentation_space = {
        # op_name: (magnitudes, signed)
        "Identity": (torch.tensor(0.0), False),
        "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (torch.linspace(0.0, 5.0, num_bins), True),
        "TranslateY": (torch.linspace(0.0, 5.0, num_bins), True),
        "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
        "Cutout": (torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
        # "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
        # "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
        # "Equalize": (torch.tensor(0.0), False),
    }
    piplines = PIPELINES.build(
        dict(type='SNNAugment', keys=['img'], augmentation_space=augmentation_space)
    )
    print(piplines)
    inputs = dict(
        img=torch.randn((16, 2, 128, 128))
    )
    out = piplines(inputs)
    print(out['img'].shape)

    for i in range(100):
        inputs = dict(
            img=torch.randn((16, 2, 128, 128))
        )
        out = piplines(inputs)
        print(out['img'].shape)



