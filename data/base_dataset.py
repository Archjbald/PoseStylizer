import random

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Resize(opt.fineSize, transforms.InterpolationMode.BICUBIC))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Resize(opt.fineSize, transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_random_trans(size):
    # degree, translate, scale, shears
    rs_params = transforms.RandomAffine.get_params(img_size=size, degrees=(-20, 40), translate=(0.1, 0.1),
                                                   scale_ranges=(0.8, 0.95), shears=None)

    def rs_transform_img(img):
        is_array = isinstance(img, np.ndarray)
        if is_array:
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img)
            if img.dtype == torch.float16:
                img = img.float()
        img = transforms.functional.affine(
            img,
            angle=rs_params[0],
            translate=rs_params[1],
            scale=rs_params[2],
            shear=rs_params[3]
        )
        if is_array:
            img = img.numpy()
            img = img.transpose((1, 2, 0))

        return img

    return rs_transform_img


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
