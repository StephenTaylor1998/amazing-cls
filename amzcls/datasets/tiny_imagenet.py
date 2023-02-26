import copy
import glob
import os.path
import cv2
import numpy as np
from mmcls.datasets import BaseDataset
from torchvision.io import read_image

from .builder import DATASETS

__all__ = ['TinyImageNet']


@DATASETS.register_module()
class TinyImageNet(BaseDataset):
    data_infos = None
    pipeline = None

    def __init__(self, data_prefix, test_mode, *args, **kwargs):
        data_prefix = data_prefix[:-1] if data_prefix.endswith('/') else data_prefix
        id_dict = load_id(os.path.join(data_prefix, 'wnids.txt'))
        filenames = glob.glob(os.path.join(
            data_prefix + ("/val/images/*.JPEG" if test_mode else "/train/*/*/*.JPEG")
        ))
        with open(os.path.join(data_prefix + '/val/val_annotations.txt'), 'r') as file:
            cls_dic = {line.split('\t')[0]: id_dict[line.split('\t')[1]] for line in file}

        self.data_infos = []
        for index in range(len(filenames)):
            img_path = filenames[index]
            if test_mode:
                label = cls_dic[img_path.split('/')[-1]]
            else:
                label = id_dict[img_path.split('/')[-3]]

            self.data_infos.append({
                'img_prefix': '',
                'img_info': {'filename': img_path},
                'gt_label': np.array(label, dtype=np.int64)})
            # self.data_infos.append({'img_prefix': img_path, 'gt_label': label})

        super(TinyImageNet, self).__init__(data_prefix=data_prefix, test_mode=test_mode, *args, **kwargs)

    def load_annotations(self):
        print(f'[INFO] [AMZCLS] Loading {"testing" if self.test_mode else "training"} annotations...')
        return self.data_infos

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])

        # data_info['img'] =
        # data_info['img'] = read_image(data_info['img'])
        return self.pipeline(data_info)


def load_id(path='./data/tiny-imagenet-200/wnids.txt'):
    id_dict = {}
    for i, line in enumerate(open(path, 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
