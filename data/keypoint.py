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


class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase)  # person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K')  # keypoints

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        # self.opt.random = True
        if self.opt.phase == 'train' or self.opt.random:
            index = random.randint(0, self.size - 1)

        P1_name, P2_name = self.pairs[index]

        if self.opt.debug:
            # while 'Transpolis' not in P1_name and 'Laure' not in P1_name and 'Chama' not in P1_name:
            while 'Transpolis' in P1_name or 'Laure' in P1_name or 'Chama' in P1_name:
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

        # P1_img = Image.new('RGB', P1_img.size)
        # P2_img = Image.new('RGB', P2_img.size)
        # BP2_img = np.empty_like(BP2_img)
        # use flip
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
                BP2 = trans(BP2)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
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
