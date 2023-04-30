import copy
from itertools import combinations

import numpy as np
from mmcls.datasets import BaseDataset

from .builder import DATASETS

__all__ = ['TimeSeqDataset']


@DATASETS.register_module()
class TimeSeqDataset(BaseDataset):
    pipeline = None
    CLASSES = ['class_t2b', 'class_l2r', 'class_b2t', 'class_r2l',
               'class_tl2br', 'class_tr2bl', 'class_bl2tr', 'class_br2tl']

    def __init__(self, data_prefix, time, sample_time, h, w, test_mode, test_rate=0.5, *args, **kwargs):
        global indices
        dataset = generate_dataset(time, h, w)
        indices = generate_indices(time, sample_time)
        test_samples = int(len(indices) * test_rate)
        self.indices = indices[test_samples:] if test_rate else indices[:test_samples]
        self.data_infos = []
        for class_label, class_data in enumerate(dataset):
            for sample_index in self.indices:
                self.data_infos.append({
                    'img': np.array(class_data[np.array(sample_index)], dtype=np.float32),
                    'gt_label': np.array(class_label, dtype=np.int64)
                })
            super(TimeSeqDataset, self).__init__(data_prefix=data_prefix, test_mode=test_mode, *args, **kwargs)

    def load_annotations(self):
        print('[INFO] [AMZCLS] Loading annotations...')
        return self.data_infos

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(data_info)


class_t2b, class_l2r, class_b2t, class_r2l = None, None, None, None
class_tl2br, class_tr2bl, class_bl2tr, class_br2tl = None, None, None, None
indices = None


def generate_dataset(time, h, w):
    global class_t2b, class_l2r, class_b2t, class_r2l
    global class_tl2br, class_tr2bl, class_bl2tr, class_br2tl
    assert time == h == w, f"[INFO] [AMZCLS] `TIME({time}) == H({h}) == W({w})`."
    if (class_t2b is None) or (class_l2r is None) or (class_b2t is None) or (class_r2l is None):
        bg = np.zeros((time, h, w))
        class_t2b = np.expand_dims(bg + np.expand_dims(np.eye(h), axis=2), axis=1)  # top to bottom
        class_l2r = np.expand_dims(bg + np.expand_dims(np.eye(h), axis=1), axis=1)  # left to right
        class_b2t = np.expand_dims(bg + np.expand_dims(np.eye(h), axis=2)[::-1], axis=1)  # bottom to top
        class_r2l = np.expand_dims(bg + np.expand_dims(np.eye(h), axis=1)[::-1], axis=1)  # right to left
        # class_tl2br = np.expand_dims(
        #     bg + np.expand_dims(np.eye(16), axis=2) @
        #     np.expand_dims(np.eye(16), axis=1), axis=1)
        # class_tr2bl = np.expand_dims(
        #     bg + np.expand_dims(np.eye(16), axis=2) @
        #     np.expand_dims(np.eye(16), axis=1)[::-1], axis=1)
        # class_bl2tr = np.expand_dims(
        #     bg + np.expand_dims(np.eye(16), axis=2)[::-1] @
        #     np.expand_dims(np.eye(16), axis=1), axis=1)
        # class_br2tl = np.expand_dims(
        #     bg + np.expand_dims(np.eye(16), axis=2)[::-1] @
        #     np.expand_dims(np.eye(16), axis=1)[::-1], axis=1)
    return class_t2b, class_l2r, class_b2t, class_r2l
    # return class_t2b, class_l2r, class_b2t, class_r2l, class_tl2br, class_tr2bl, class_bl2tr, class_br2tl


def generate_indices(time, sample_time):
    global indices
    assert time >= sample_time, f"[INFO] [AMZCLS] `TIME({time}) >= SAMPLE_TIME({sample_time})`."
    if indices is None:
        indices = list(combinations(range(time), sample_time))
    return indices
