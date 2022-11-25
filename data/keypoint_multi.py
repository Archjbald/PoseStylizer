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
        phase = opt.phase
        if phase == 'val':
            opt.phase = 'test'
            opt.ratio_multi = 1 / 5
        for is_real, (root, pairLst, custom_transform) in enumerate([
            ('./dataset/synthe_dripe/', './dataset/synthe_dripe/synthe-pairs-{}.csv',
             DraiverTransform(equalize=opt.equalize, rotate_angle=42)),
            ('./dataset/draiver_data/', './dataset/draiver_data/draiver-pairs-{}.csv',
             DraiverTransform(equalize=opt.equalize, color_swap=opt.color_swap if opt.phase == 'train' else False))
        ]):
            opt_set = Namespace(**vars(opt))
            opt_set.dataroot = root
            opt_set.pairLst = pairLst.format(opt.phase)
            self.datasets.append(KeyDataset())
            self.datasets[-1].initialize(opt_set, custom_transform=custom_transform)
            self.total_size += self.datasets[-1].size
            self.idxs += [(len(self.datasets) - 1, i) for i in range(self.datasets[-1].size)]

        ratios = [data.size / self.total_size for data in self.datasets]
        # if self.opt.phase == 'train':
        synthe_size = self.datasets[0].size
        real_size = sum([dataset.size for dataset in self.datasets[1:]])
        if (1 - opt.ratio_multi) < ratios[0]:
            self.idxs = [(0, i) for i in random.sample(range(self.datasets[0].size),
                                                       int(real_size * (1 - opt.ratio_multi) / opt.ratio_multi))]
            self.idxs += [(d + 1, i) for d, dataset in enumerate(self.datasets[1:]) for i in range(dataset.size)]
        elif (1 - opt.ratio_multi) > ratios[0]:
            self.idxs = [(0, i) for i in range(self.datasets[0].size)]
            self.idxs += [(d + 1, i) for d, dataset in enumerate(self.datasets[1:]) for i in
                          random.sample(range(dataset.size),
                                        int(synthe_size * opt.ratio_multi / (
                                                1 - opt.ratio_multi) * dataset.size / real_size))]

        self.ratios = [sum([ix[0] == i for ix in self.idxs]) / len(self.idxs) for i in range(len(self.datasets))]
        if not self.opt.debug and opt.phase == 'train':
            random.shuffle(self.idxs)
        else:
            max_size_ratios = [round(opt.max_dataset_size * ratio) for ratio in self.ratios]
            sorted_idx = []
            end_sorted_idx = []
            for k, mr in enumerate(max_size_ratios):
                sorted_idx += self.idxs[int(len(self.idxs) * sum(self.ratios[:k])):
                                        int(len(self.idxs) * sum(self.ratios[:k])) + mr]
                end_sorted_idx += self.idxs[int(len(self.idxs) * sum(self.ratios[:k])) + mr:
                                            int(len(self.idxs) * sum(self.ratios[:k + 1]))]
            self.idxs = sorted_idx + end_sorted_idx

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
    def __init__(self, equalize=False, color_swap=False, rotate_angle=0, proba=1.):
        self.equalize = equalize
        self.color_swap = color_swap
        self.rotate_angle = rotate_angle
        self.proba = proba

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
        if self.color_swap and c == 3 and random.random() <= self.proba:
            channels = [0, 1, 2]
            random.shuffle(channels)
            x = Image.fromarray(np.array(x)[:, :, channels])

        x = transforms.functional.affine(x, angle=0, translate=(0.1 * w, 0.1 * h), scale=1, shear=(0, 0))
        x = transforms.functional.rotate(x, self.rotate_angle)

        if c <= 3:
            # x = ((x + 1) * 128).to(torch.uint8)
            # x = transforms.functional.equalize(x)
            # x = Image.fromarray(x)
            pass
        else:
            x = x.moveaxis(0, -1)
            x = x.numpy()

        return x
