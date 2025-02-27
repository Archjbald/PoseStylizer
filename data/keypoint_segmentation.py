import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch

class KeySegDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints
        self.dir_S = os.path.join(opt.dataroot, opt.phase + 'SegNumpy') #segmentations

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
        if self.opt.phase == 'train' or self.opt.random:
            index = random.randint(0, self.size-1)
        
        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1
        MP1_path = os.path.join(self.dir_S, P1_name + '.npy') # segmentation of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name) # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2
        MP2_path = os.path.join(self.dir_S, P2_name + '.npy') # segmentation of person 2

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path)
        
        MP1_img = np.load(MP1_path)
        MP2_img = np.load(MP2_path)
        # use flip
        if self.opt.phase == 'train' and not self.opt.no_flip:
            flip_random = random.uniform(0,1)
            if flip_random > 0.5:
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip
                
                MP1_img = np.array(MP1_img[:, ::-1]) # flip
                MP2_img = np.array(MP2_img[:, ::-1]) # flip
                
        P1 = self.transform(P1_img)
        P2 = self.transform(P2_img)

        BP1 = torch.from_numpy(BP1_img).float() #h, w, c
        BP1 = BP1.transpose(2, 0) #c,w,h
        BP1 = BP1.transpose(2, 1) #c,h,w 

        BP2 = torch.from_numpy(BP2_img).float()
        BP2 = BP2.transpose(2, 0) #c,w,h
        BP2 = BP2.transpose(2, 1) #c,h,w 
        
        MP1 = torch.from_numpy(MP1_img).float() #h, w
        MP2 = torch.from_numpy(MP2_img).float()
        MP1 = MP1.unsqueeze(0)
        MP2 = MP2.unsqueeze(0)
        
        return {'P1': P1, 'BP1': BP1, 'MP1': MP1, 'P2': P2, 'BP2': BP2, 'MP2': MP2, 
                'P1_path': P1_name, 'P2_path': P2_name}
                

    def __len__(self):
        if self.opt.phase == 'train':
            return self.opt.epoch_size
        else:
            return self.size

    def name(self):
        return 'KeySegDataset'
