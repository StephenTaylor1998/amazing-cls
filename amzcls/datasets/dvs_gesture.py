import numpy as np
from mmcls.datasets import BaseDataset
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

from .builder import DATASETS

__all__ = ['DVSGesture']


@DATASETS.register_module()
class DVSGesture(BaseDataset):
    def __init__(self, time_step, data_type='frame', split_by='number', *args, **kwargs):
        self.time_step = time_step
        self.data_type = data_type
        self.split_by = split_by
        super(DVSGesture, self).__init__(*args, **kwargs)

    def load_annotations(self):
        print('[INFO] [AMZCLS] Loading annotations...')
        dvs_dataset = DVS128Gesture(
            root=self.data_prefix,
            train=~self.test_mode,
            data_type=self.data_type,
            frames_number=self.time_step,
            split_by=self.split_by)

        data_infos = []
        for sample in dvs_dataset.samples:
            data_infos.append({
                'img': load_npz(sample[0]),
                'gt_label': np.array(sample[1], dtype=np.int64)  # todo: 2023-01-26
            })
        return data_infos


def load_npz(file_name, data_type='frames'):
    return np.load(file_name, allow_pickle=True)[data_type].astype(np.float32)
