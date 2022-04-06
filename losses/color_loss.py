import sys

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import torchvision.models as models

from util.util import get_kps


def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: 4D tensor
    :return: 4D tensor
    """
    B, C, H, W = rgb.shape
    eps = sys.float_info.epsilon
    maxv, maxc = rgb.max(dim=1)
    minv, minc = rgb.min(dim=1)

    hsv = torch.zeros_like(rgb)
    hsv[maxc == minc, 0] = torch.zeros_like(hsv[maxc == minc, 0])
    hsv[maxc == 0, 0] = (((rgb[:, 1] - rgb[:, 2]) * 60.0 / (maxv - minv + eps)) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[:, 2] - rgb[:, 0]) * 60.0 / (maxv - minv + eps)) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[:, 0] - rgb[:, 1]) * 60.0 / (maxv - minv + eps)) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = torch.zeros_like(hsv[maxv == 0, 1])
    hsv[maxv != 0, 1] = (1 - minv / (maxv + eps))[maxv != 0]
    hsv[:, 2] = maxv

    return hsv


def rgb2lab(rgb):
    # rgb2xyz
    arr = rgb
    mask = arr > 0.04045
    arr[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
    arr[~mask] /= 12.92

    xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                                 [0.212671, 0.715160, 0.072169],
                                 [0.019334, 0.119193, 0.950227]], device=rgb.device)

    arr = arr.movedim(1, -1)
    xyz = arr @ xyz_from_rgb.T

    illuminant = "D65"
    observer = "2"

    arr = xyz

    xyz_ref_white = torch.tensor((0.95047, 1., 1.08883), device=rgb.device)

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = arr[mask] ** 1 / 3
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    lab = torch.stack([L, a, b], dim=1)
    return lab, xyz


class ColorLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.loss = nn.MSELoss()
        self.nb_patch = 6
        self.vgg = models.vgg19(pretrained=True).cuda()

    def forward(self, img_in, bp_in, img_out, bp_out):
        patches_in, patches_out = self.get_patches(img_in, bp_in, img_out, bp_out)
        feats_in = self.vgg(patches_in)
        feats_out = self.vgg(patches_out)
        loss = self.loss(feats_out.mT @ feats_out, feats_in.mT @ feats_in) * self.opt.lambda_patch
        return loss / self.nb_patch

    def get_patches(self, img_1, bp_1, img_2, bp_2, concat=True):
        height = img_1.shape[-2]
        h_patch = height // self.nb_patch
        thresh = [h_patch * i for i in range(self.nb_patch + 1)]
        # thresh = [0] + [h_patch * (i + 1) for i in range(self.nb_patch - 1)] + [height]
        patches = [[img[..., t:thresh[i + 1], :] for i, t in enumerate(thresh[:-1])] for img in [img_1, img_2]]
        if concat:
            patches = [torch.concat(p, dim=0) for p in patches]
        return patches


class ColorLossPatch(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.loss = nn.MSELoss()
        self.patch_size = 15

    def forward(self, img_in, bp_in, img_out, bp_out):
        patches_in, patches_out = self.get_patches(img_in, bp_in, img_out, bp_out)
        loss = self.loss(patches_out, patches_in) * self.opt.lambda_patch
        return loss

    def get_patches(self, img_1, bp_1, img_2, bp_2):
        """
        Generate list of patch ids according to skeleton.
        Skeleton ids are shared across scales, random are not
        :return: list of ids per scales, num_patches

        Parameters
        ----------
        img_1 image_1 (Bx3xHxW)
        bp_1 skeleton_1 (BxCxHxW)
        img_2 image_1 (Bx3xHxW)
        bp_2 skeleton_2 (BxCxHxW)
        """
        device = bp_1.device

        patch_shape = (self.patch_size, self.patch_size)
        patch_size = patch_shape[0] * patch_shape[1]

        B, C, H, W = bp_1.shape
        kps_1, v_1 = get_kps(bp_1, thresh=0.5)
        kps_2, v_2 = get_kps(bp_2, thresh=0.5)
        v = v_1 * v_2

        # Padding
        pad_h = patch_shape[0] // 2
        pad_w = patch_shape[1] // 2
        h_pad = H + 2 * pad_h
        w_pad = W + 2 * pad_w

        # Patches
        idx = torch.arange(0, h_pad * w_pad).view(h_pad, w_pad)
        idx_patches = idx.unfold(0, patch_shape[0], 1).unfold(1, patch_shape[1], 1).contiguous().view(H, W, -1)
        idx_patches = torch.stack([torch.div(idx_patches, w_pad, rounding_mode='trunc'), idx_patches % w_pad], dim=-2)

        ret_patches = []
        for img, kps in [(img_1, kps_1), (img_2, kps_2)]:
            patches = torch.zeros((B, C, patch_size, 3), device=device, dtype=img.dtype)
            img_pad = F.pad(img, (pad_w, pad_w, pad_h, pad_h), value=-1)
            for b in range(B):
                for k in range(C):
                    patches[b, k] = (img_pad[b, :, idx_patches[kps[b, k, 0], kps[b, k, 1]][0],
                                     idx_patches[kps[b, k, 0], kps[b, k, 1]][1]] * v[b, k]).transpose(0, 1)
            ret_patches.append(patches)

        return ret_patches
