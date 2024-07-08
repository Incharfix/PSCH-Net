# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:21:32
# @Email:  cshzxie@gmail.com

import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import open3d as o3d
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO

label_mapping = {
    3: '03001627',
    6: '04379243',
    5: '04256520',
    1: '02933112',
    4: '03636649',
    2: '02958343',
    0: '02691156',
    7: '04530566'
}

@unique
class DatasetSubset(Enum):
    TRAIN = 0
    VAL = 2
    TEST = 1



def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data

code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}

def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            # file_path = sample['%s_path' % ri]
            file_path = sample['%s' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            # print(file_path)
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data



class Completion3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

#   partial_cloud_path  gtcloud_path   partial_dense             partial_input   partial_bilinear  drop_dense
    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half'] if subset == DatasetSubset.TEST else ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half']

        return Dataset({
            'required_items': required_items,
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                   'n_points': cfg.CONST.N_INPUT_POINTS
               },
                'objects': ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half']
            },
                {
                    'callback': 'ScalePoints',
                    'parameters': {
                        'scale': 0.95
                    },
                    'objects': ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half']
                },
                {
                'callback': 'ToTensor',
                'objects': ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half']
            }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback': 'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },

                'objects': ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half']
            },{
                'callback': 'ToTensor',
                'objects': ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_dense', 'partial_input','partial_bilinear', 'drop_dense','gt_half']
            }])

        # print(11111111)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,

   # 临时保留名字  partial_cloud_path  gtcloud_path  ## partial_dense             partial_dense   partial_bilinear  drop_dense
                    'partial_dense':    cfg.DATASETS.COMPLETION3D.partial_dense % (subset, dc['taxonomy_id'], s),
                    'partial_input':    cfg.DATASETS.COMPLETION3D.partial_input % (subset, dc['taxonomy_id'], s),
                    'partial_bilinear': cfg.DATASETS.COMPLETION3D.partial_bilinear % (subset, dc['taxonomy_id'], s),
                    'drop_dense':       cfg.DATASETS.COMPLETION3D.drop_dense % (subset, dc['taxonomy_id'], s),
                    'gt_half':          cfg.DATASETS.COMPLETION3D.gt_half % (subset, dc['taxonomy_id'], s),

                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list



# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader
    # 'Completion3DPCCT': Completion3DPCCTDataLoader,
    # 'ShapeNet': ShapeNetDataLoader,
    # 'ShapeNetCars': ShapeNetCarsDataLoader,
    # 'KITTI': KittiDataLoader
}  # yapf: disable



