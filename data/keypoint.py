import math
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_random_trans
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms

from data.pose_transform import make_gaussian_limb_masks


class KeyDataset(BaseDataset):
    def initialize(self, opt, custom_transform=None):
        self.opt = opt
        self.root = opt.dataroot

        if opt.phase == 'val':
            opt.phase = 'test'
        self.dir_P = os.path.join(opt.dataroot, opt.phase)  # person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K')  # keypoints

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)
        self.custom_transform = custom_transform if custom_transform else lambda x, name: x

    def init_categories(self, pairLst, annoLst=None):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)
            if self.opt.debug and i > 1000:
                break
        self.size = len(self.pairs)

        if self.opt.phase == 'train' and not self.opt.debug:
            random.shuffle(self.pairs)
        elif self.opt.random:
            rng = random.Random(31415)
            rng.shuffle(self.pairs)

        print(f"Loaded {len(self.pairs)} pairs")

        if annoLst is not None:
            print('Loading data annos ...')
            annotations_file = pd.read_csv(annoLst, sep=':')
            self.annos = annotations_file.set_index('name')
            print('Loading data annos finished ...')
        else:
            self.annos = None

    def __getitem__(self, index):
        # self.opt.random = True
        if self.opt.phase == 'train' or self.opt.random:
            index = random.randint(0, self.size - 1)

        P1_name, P2_name = self.pairs[index]

        P1_path = os.path.join(self.dir_P, P1_name)  # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy')  # bone of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name)  # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy')  # bone of person 2

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path)  # h, w, c
        BP2_img = np.load(BP2_path)

        THRESHOLD = 8 * 0.013 * BP1_img.shape[0] ** 2  # ~ 880 * 8
        if self.opt.phase == 'train' or self.opt.random:
            if BP1_img.sum() < THRESHOLD or BP2_img.sum() < THRESHOLD:
                # get a new img
                return self.__getitem__(index)

        # P1_img = Image.new('RGB', P1_img.size)
        # P2_img = Image.new('RGB', P2_img.size)
        # BP2_img = np.empty_like(BP2_img)
        # use flip

        P1_img = self.custom_transform(P1_img, P1_name)
        P2_img = self.custom_transform(P2_img, P2_name)
        BP1_img = self.custom_transform(BP1_img, P1_name)
        BP2_img = self.custom_transform(BP2_img, P2_name)

        if self.opt.phase == 'train':
            # print ('use_flip ...')
            flip_random = random.uniform(0, 1)
            rs_random = random.uniform(0, 1)
            if rs_random > 0.6 and not self.opt.no_rotate:
                rs_transform = get_random_trans(P1_img.size[:2])

                P1_img = rs_transform(P1_img)
                P2_img = rs_transform(P2_img)

                BP1_img = rs_transform(BP1_img)
                BP2_img = rs_transform(BP2_img)

            if flip_random > 0.5 and not self.opt.no_flip:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = flip_keypoints(BP1_img)  # flip
                BP2_img = flip_keypoints(BP2_img)  # flip

        BP1 = torch.from_numpy(BP1_img).float()  # h, w, c
        BP1 = BP1.transpose(2, 0)  # c,w,h
        BP1 = BP1.transpose(2, 1)  # c,h,w

        BP2 = torch.from_numpy(BP2_img).float()
        BP2 = BP2.transpose(2, 0)  # c,w,h
        BP2 = BP2.transpose(2, 1)  # c,h,w

        P1 = self.transform(P1_img)
        P2 = self.transform(P2_img)

        if not P1.shape[-2:] == BP1.shape[-2:]:
            trans = transforms.Resize((P1.shape[-2], P1.shape[-1]))
            with torch.no_grad():
                BP1 = trans(BP1)

        if not P2.shape[-2:] == BP2.shape[-2:]:
            trans = transforms.Resize((P2.shape[-2], P2.shape[-1]))
            with torch.no_grad():
                BP2 = trans(BP2)

        height = P1.shape[-2]
        width = P1.shape[-1]
        scale_factor = 2 ** max(1, self.opt.G_n_downsampling, self.opt.D_n_downsampling)
        height_reduced = height / scale_factor
        width_reduced = width / scale_factor
        if not self.opt.dataset == 'fashion' and (not height_reduced.is_integer() or not width_reduced.is_integer()):
            height_target = 32 * math.ceil(height_reduced)
            width_target = 32 * math.ceil(width_reduced)
            pad_top = (height_target - height) // 2
            pad_bot = height_target - height - pad_top
            pad_left = (width_target - width) // 2
            pad_right = width_target - width - pad_left
            P1 = torch.nn.functional.pad(input=P1, pad=(pad_left, pad_right, pad_top, pad_bot))
            P2 = torch.nn.functional.pad(input=P2, pad=(pad_left, pad_right, pad_top, pad_bot))
            BP1 = torch.nn.functional.pad(input=BP1, pad=(pad_left, pad_right, pad_top, pad_bot))
            BP2 = torch.nn.functional.pad(input=BP2, pad=(pad_left, pad_right, pad_top, pad_bot))

        # if self.opt.phase == 'train' and 'SSIM' in self.opt.L1_type:
        if 'L1_type' in self.opt and 'SSIM' in self.opt.L1_type:
            BP2_mask = make_gaussian_limb_masks(BP2)
            BP2_mask = torch.from_numpy(BP2_mask).float()
        else:
            BP2_mask = []

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'BP2_mask': BP2_mask,
                'P1_path': P1_name, 'P2_path': P2_name}

    def __len__(self):
        if self.opt.phase == 'train':
            return self.opt.epoch_size
        else:
            return self.size

    def name(self):
        return 'KeyDataset'


def flip_keypoints(bp):
    bp = np.array(bp[:, ::-1, :])
    idxs = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16] + list(range(18, bp.shape[-1]))
    bp = bp[:, :, idxs]
    return bp
