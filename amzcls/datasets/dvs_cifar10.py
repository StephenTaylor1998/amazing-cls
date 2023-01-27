import numpy as np
from mmcls.datasets import BaseDataset
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

from .builder import DATASETS

__all__ = ['DVSCifar10']


@DATASETS.register_module()
class DVSCifar10(BaseDataset):
    def __init__(self, time_step, data_type='frame', split_by='number', *args, **kwargs):
        self.time_step = time_step
        self.data_type = data_type
        self.split_by = split_by
        super(DVSCifar10, self).__init__(*args, **kwargs)

    def load_annotations(self):
        print('[INFO] [AMZCLS] Loading annotations...')
        dvs_dataset = CIFAR10DVS(
            root=self.data_prefix,
            data_type=self.data_type,
            frames_number=self.time_step,
            split_by=self.split_by)
        train_set, test_set = split_to_train_test_set(0.9, dvs_dataset, 10)
        if self.test_mode:
            dvs_dataset = test_set
        else:
            dvs_dataset = train_set

        data_infos = []
        for index in dvs_dataset.indices:
            sample = dvs_dataset.dataset[index]
            data_infos.append({
                'img': sample[0],
                'gt_label': np.array(sample[1], dtype=np.int64)  # todo: 2023-01-26
            })

        return data_infos


def load_npz(file_name, data_type='frames'):
    return np.load(file_name, allow_pickle=True)[data_type].astype(np.float32)
