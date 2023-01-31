import copy

import numpy as np
from mmcls.datasets import BaseDataset
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

from .builder import DATASETS

__all__ = ['DVSGesture']


@DATASETS.register_module()
class DVSGesture(BaseDataset):
    pipeline = None

    def __init__(self, data_prefix, test_mode, time_step, data_type='frame', split_by='number', *args, **kwargs):
        self.time_step = time_step
        self.data_type = data_type
        self.split_by = split_by
        dvs_dataset = load_dvs_gesture(data_prefix, test_mode, time_step, data_type, split_by)
        self.data_infos = []
        for sample in dvs_dataset.samples:
            self.data_infos.append({
                'img': sample[0],
                'gt_label': np.array(sample[1], dtype=np.int64)  # todo: 2023-01-26
            })
        super(DVSGesture, self).__init__(data_prefix=data_prefix, test_mode=test_mode, *args, **kwargs)

    def load_annotations(self):
        print('[INFO] [AMZCLS] Loading annotations...')
        return self.data_infos

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        data_info['img'] = load_npz(data_info['img'])
        return self.pipeline(data_info)


def load_npz(file_name, data_type='frames'):
    return np.load(file_name, allow_pickle=True)[data_type].astype(np.float32)


dvs_train_dataset = None
dvs_test_dataset = None


def load_dvs_gesture(data_prefix, test_mode, time_step, data_type='frame', split_by='number'):
    if test_mode:
        global dvs_test_dataset
        if dvs_test_dataset is None:
            dvs_test_dataset = DVS128Gesture(
                root=data_prefix,
                train=False,
                data_type=data_type,
                frames_number=time_step,
                split_by=split_by)
            print(f'[INFO] [AMZCLS] Processing testing dataset...')
        return dvs_test_dataset
    else:
        global dvs_train_dataset
        if dvs_train_dataset is None:
            dvs_train_dataset = DVS128Gesture(
                root=data_prefix,
                train=True,
                data_type=data_type,
                frames_number=time_step,
                split_by=split_by)
            print(f'[INFO] [AMZCLS] Processing training dataset...')
        return dvs_train_dataset
