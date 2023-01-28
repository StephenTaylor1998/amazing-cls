import math
import os.path
import pickle
import numpy as np
import torch
from mmcls.datasets import BaseDataset
# from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from tqdm import tqdm

from .builder import DATASETS

__all__ = ['DVSCifar10']


@DATASETS.register_module()
class DVSCifar10(BaseDataset):
    data_infos = None

    def __init__(self, data_prefix, test_mode, time_step, data_type='frame',
                 split_by='number', use_ckpt=False, *args, **kwargs):
        self.time_step = time_step
        self.data_type = data_type
        self.split_by = split_by
        dvs_dataset = load_dvs_cifar10(data_prefix, test_mode, time_step, data_type, split_by)
        dataset_indices = dvs_cifar10_indices(train_ratio=0.9, test_mode=test_mode)
        if use_ckpt:
            ckpt = create_dir(data_prefix, test_mode, time_step, data_type, split_by)
            self.data_infos = load_data_infos_with_ckpt(dvs_dataset, dataset_indices, ckpt)
        else:
            self.data_infos = load_data_infos(dvs_dataset, dataset_indices)
        super(DVSCifar10, self).__init__(data_prefix=data_prefix, test_mode=test_mode, *args, **kwargs)

    def load_annotations(self):
        print(f'[INFO] [AMZCLS] Loading {"testing" if self.test_mode else "training"} annotations...')
        return self.data_infos


dvs_dataset = None


def load_dvs_cifar10(data_prefix, test_mode, time_step, data_type='frame', split_by='number'):
    global dvs_dataset
    if dvs_dataset is None:
        dvs_dataset = CIFAR10DVS(
            root=data_prefix,
            data_type=data_type,
            frames_number=time_step,
            split_by=split_by)
        print(f'[INFO] [AMZCLS] Processing {"testing" if test_mode else "training"} dataset...')
    return dvs_dataset


def dvs_cifar10_indices(train_ratio: float, test_mode):
    # indices = [[i for i in range(j * 1000, (j + 1) * 1000)] for j in range(10)]
    train_indices = [[i for i in range(j * 1000, j * 1000 + int(1000 * train_ratio))] for j in range(10)]
    test_indices = [[i for i in range(j * 1000 + int(1000 * train_ratio), (j + 1) * 1000)] for j in range(10)]
    return test_indices if test_mode else train_indices


# def load_npz(file_name, data_type='frames'):
#     return np.load(file_name, allow_pickle=True)[data_type].astype(np.float32)


# implement without checkpoint
def load_data_infos(dvs_dataset, dataset_indices, unpack=True):
    sort_by_class = []
    for class_indices in dataset_indices:
        class_infos = []
        for index in tqdm(class_indices):
            sample = dvs_dataset.samples[index]
            class_infos.append({
                'img': np.load(sample[0], allow_pickle=True)['frames'].astype(np.float32),
                'gt_label': np.array(sample[1], dtype=np.int64)
            })
        sort_by_class.append(class_infos)
    del dvs_dataset

    if unpack:
        data_infos = []
        for data in sort_by_class:
            data_infos.extend(data)
    else:
        data_infos = sort_by_class
    return data_infos


# implement with checkpoint
def load_data_infos_with_ckpt(dvs_dataset, dataset_indices, ckpt):
    sort_by_class = _load_ckpt(ckpt)
    if sort_by_class is None:
        sort_by_class = load_data_infos(dvs_dataset, dataset_indices, unpack=False)
        _save_ckpt(sort_by_class, ckpt)

    del dvs_dataset
    data_infos = []
    for data in sort_by_class:
        data_infos.extend(data)
    return data_infos


def _load_ckpt(dir: str):
    if os.path.exists(dir):
        try:
            print(f'[INFO] [AMZCLS] Loading dataset from ckpt `{dir}`.')
            sort_by_class = []
            for index in tqdm(range(10)):
                path = os.path.join(dir, f"{index}.pth")
                sort_by_class.append(torch.load(path))
            return sort_by_class
        except:
            print('[INFO] [AMZCLS] Skip loading ckpt ...')
    return None


def _save_ckpt(data: list, dir: str):
    for index, d in enumerate(data):
        path = os.path.join(dir, f"{index}.pth")
        print(f'[INFO] [AMZCLS] Saving ckpt to `{path}` ...')
        torch.save(data[index], path)


def create_dir(data_prefix, test_mode, time_step, data_type, split_by):
    dir_name = os.path.join(
        data_prefix,
        f"{'Test' if test_mode else 'Train'}-Time{time_step}-{data_type}-{split_by}"
    )
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    return dir_name
