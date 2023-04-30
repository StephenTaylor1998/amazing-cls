import torch

from amzcls.datasets import PIPELINES

if __name__ == '__main__':
    piplines = PIPELINES.build(dict(type='SNNAugment', keys=['img']))
    print(piplines)
    inputs = dict(
        img=torch.randn((4, 2, 128, 128))
    )
    out = piplines(inputs)
    print(out)


