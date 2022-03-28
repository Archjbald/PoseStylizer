import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as fn

from util.util import get_kps


class ColorLoss(nn.Module):
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
