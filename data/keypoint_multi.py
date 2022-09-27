import math
import os.path
from PIL import Image
from argparse import Namespace
import random
import numpy as np
import torch
import torchvision.transforms as transforms

from data.base_dataset import BaseDataset
from data.keypoint import KeyDataset
from data.pose_transform import make_gaussian_limb_masks


class KeyDatasetMulti(BaseDataset):
    def __init__(self):
        super(KeyDatasetMulti, self).__init__()
        self.datasets = []
        self.ratios = []
        self.idxs = []
        self.len_total = 0

    def initialize(self, opt):
        self.opt = opt
        for root, pairLst in [('./dataset/synthe_dripe/', './dataset/synthe_dripe/synthe-pairs-train.csv'),
                              ('./dataset/draiver_data/', './dataset/draiver_data/draiver-pairs-train.csv')]:
            opt_set = Namespace(**vars(opt))
            opt_set.dataroot = root
            opt_set.pairLst = pairLst
            self.datasets.append(KeyDataset())
            self.datasets[-1].initialize(opt_set)
            self.len_total += len(self.datasets[-1])
            self.idxs += [(len(self.datasets) - 1, i) for i in range(len(self.datasets))]

        self.ratios = [len(data) / self.len_total for data in self.datasets]
        if self.opt.phase == 'train' and not self.opt.debug:
            random.shuffle(self.idxs)

    def __getitem__(self, index):
        idx = self.idxs[index]
        return self.datasets[idx[0]][idx[1]]

    def name(self):
        return 'KeyDatasetMulti'

    def __len__(self):
        return self.len_total


def flip_keypoints(bp):
    bp = np.array(bp[:, ::-1, :])
    idxs = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16] + list(range(18, bp.shape[-1]))
    bp = bp[:, :, idxs]
    return bp
