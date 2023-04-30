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

    piplines = PIPELINES.build(
        dict(type='TimeSample', keys=['img'], time_step=16, sample_step=12, use_rand=True)
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



