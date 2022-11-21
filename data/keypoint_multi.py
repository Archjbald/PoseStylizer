import math
import os.path
from PIL import Image
from argparse import Namespace
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional

from data.base_dataset import BaseDataset
from data.keypoint import KeyDataset, flip_keypoints
from data.pose_transform import make_gaussian_limb_masks


class KeyDatasetMulti(BaseDataset):
    def __init__(self):
        super(KeyDatasetMulti, self).__init__()
        self.datasets = []
        self.ratios = []
        self.idxs = []
        self.total_size = 0

    def initialize(self, opt):
        self.opt = opt
        for is_real, (root, pairLst, custom_transform) in enumerate([
            ('./dataset/synthe_dripe/', './dataset/synthe_dripe/synthe-pairs-{}.csv',
             transforms.functional.equalize if opt.equalize else None),
            ('./dataset/draiver_data/', './dataset/draiver_data/draiver-pairs-{}.csv',
             DraiverTransform(equalize=opt.equalize))
        ]):
            opt_set = Namespace(**vars(opt))
            opt_set.dataroot = root
            opt_set.pairLst = pairLst.format(opt.phase)
            self.datasets.append(KeyDataset())
            self.datasets[-1].initialize(opt_set, custom_transform=custom_transform)
            self.total_size += self.datasets[-1].size
            self.idxs += [(len(self.datasets) - 1, i) for i in range(self.datasets[-1].size)]

        self.ratios = [data.size / self.total_size for data in self.datasets]
        # if self.opt.phase == 'train':
        synthe_size = self.datasets[0].size
        real_size = sum([dataset.size for dataset in self.datasets[1:]])
        if (1 - opt.ratio_multi) < self.ratios[0]:
            self.idxs = [(0, i) for i in random.sample(range(self.datasets[0].size),
                                                       int(real_size * (1 - opt.ratio_multi) / opt.ratio_multi))]
            self.idxs += [i for dataset in self.datasets[1:] for i in range(dataset.size)]
        elif (1 - opt.ratio_multi) > self.ratios[0]:
            self.idxs = [(0, i) for i in range(self.datasets[0].size)]
            self.idxs += [(d + 1, i) for d, dataset in enumerate(self.datasets[1:]) for i in
                          random.sample(range(dataset.size),
                                        int(synthe_size * opt.ratio_multi / (
                                                1 - opt.ratio_multi) * dataset.size / real_size))]
        if not self.opt.debug:
            random.shuffle(self.idxs)

        print(f'Multi dataset : loaded {len(self.idxs)} pairs')

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.idxs[index]
        return self.datasets[index[0]][index[1]]

    def name(self):
        return 'KeyDatasetMulti'

    def __len__(self):
        if self.opt.phase == 'train':
            return self.opt.epoch_size
        else:
            return self.total_size


class DraiverTransform:
    def __init__(self, equalize):
        self.equalize = equalize

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            h, w, c = x.shape
            x = torch.Tensor(x)
            x = x.moveaxis(-1, 0)
        else:
            w, h = x.size
            c = len(x.mode)

        if self.equalize and c <= 3:
            x = transforms.functional.equalize(x)
        x = transforms.functional.affine(x, angle=0, translate=(0.1 * w, 0.1 * h), scale=1, shear=(0, 0))
        x = transforms.functional.rotate(x, -42)

        if c <= 3:
            # x = ((x + 1) * 128).to(torch.uint8)
            # x = transforms.functional.equalize(x)
            # x = Image.fromarray(x)
            pass
        else:
            x = x.moveaxis(0, -1)
            x = x.numpy()

        return x
