import os.path

import numpy as np
import torch
from mmcls.datasets import BaseDataset
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from tqdm import tqdm

from .builder import DATASETS

__all__ = ['DVSCifar10']

train_set = None
test_set = None


@DATASETS.register_module()
class DVSCifar10(BaseDataset):
    data_infos = None
    def __init__(self, data_prefix, test_mode, time_step, data_type='frame',
                 split_by='number', use_checkpoint=False, *args, **kwargs):
        self.time_step = time_step
        self.data_type = data_type
        self.split_by = split_by
        if use_checkpoint:
            self.data_infos = _load_data_checkpoint(data_prefix, test_mode, time_step, data_type, split_by)
        if self.data_infos is None:
            self.data_infos = load_dvs_cifar10(
                data_prefix, test_mode, time_step, data_type, split_by, use_checkpoint)
        super(DVSCifar10, self).__init__(data_prefix=data_prefix, test_mode=test_mode, *args, **kwargs)

    def load_annotations(self):
        print(f'[INFO] [AMZCLS] Loading {"testing" if self.test_mode else "training"} annotations...')
        return self.data_infos


def _load_data_checkpoint(path: str, test_mode, time_step, data_type, split_by):
    path = os.path.join(path, _file_name(test_mode, time_step, data_type, split_by))
    if os.path.isfile(path):
        try:
            print(f'[INFO] [AMZCLS] Loading dataset from checkpoint `{path}`.')
            return torch.load(path)
        except:
            print('[INFO] [AMZCLS] File error...')
    return None


def _save_data_checkpoint(data, path: str, test_mode, time_step, data_type, split_by):
    path = os.path.join(path, _file_name(test_mode, time_step, data_type, split_by))
    torch.save(data, path)


def load_dvs_cifar10(
        data_prefix, test_mode, time_step, data_type='frame', split_by='number', use_checkpoint=False):
    global train_set
    global test_set
    if (train_set and test_set) is None:
        dvs_dataset = CIFAR10DVS(
            root=data_prefix,
            data_type=data_type,
            frames_number=time_step,
            split_by=split_by)
        train_set, test_set = split_to_train_test_set(0.9, dvs_dataset, 10)

    dvs_dataset = test_set if test_mode else train_set
    print(f'[INFO] [AMZCLS] Processing {"testing" if test_mode else "training"} dataset...')
    data_infos = []
    for index in tqdm(dvs_dataset.indices):
        sample = dvs_dataset.dataset[index]
        data_infos.append({
            'img': sample[0],
            'gt_label': np.array(sample[1], dtype=np.int64)  # todo: 2023-01-26
        })
    if use_checkpoint:
        path = os.path.join(data_prefix, _file_name(test_mode, time_step, data_type, split_by))
        print(f'[INFO] [AMZCLS] Saving dataset to checkpoint `{path}`.')
        _save_data_checkpoint(data_infos, data_prefix, test_mode, time_step, data_type, split_by)
    return data_infos


def _file_name(test_mode, time_step, data_type, split_by):
    return f"{'Test' if test_mode else 'Train'}-Time{time_step}-{data_type}-{split_by}"
