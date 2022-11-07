# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .modules import DeconvStage, PoseNet
from .resnet import ResNet, BasicBlock, Bottleneck

logger = logging.getLogger(__name__)


class PoseResNet(PoseNet):
    def __init__(self, block, layers, gen_final=True):
        super(PoseResNet, self).__init__()

        self.resnet = ResNet(block, layers)
        self.final_stage = DeconvStage(self.resnet.inplanes)
        self.gen_final_train = gen_final
        self.gen_final = gen_final
        self.mesh_grid = None

        self.img_size = (192, 256)

    def switch_gen(self):
        self.gen_final = self.gen_final_train or not self.gen_final

    def get_feature_extractor(self, perceptual_layers, gpu_ids):
        submodel = nn.Sequential(

        )
        for i, (name, layer) in enumerate(self.resnet.named_children()):
            submodel.add_module(name, layer)
            if i == perceptual_layers:
                break
        submodel = torch.nn.DataParallel(submodel, device_ids=gpu_ids).cuda()
        return submodel

    def resize(self, x):
        target_ratio = self.img_size[0] / self.img_size[1]
        actual_ratio = x.shape[-2] / x.shape[-1]
        pad_left, pad_right, pad_top, pad_bot = (0, 0, 0, 0)
        if target_ratio > actual_ratio:
            x_scaled = nn.functional.interpolate(x, (round(self.img_size[1] * actual_ratio), self.img_size[1]))
            pad_top = (self.img_size[0] - x_scaled.shape[-2]) // 2
            pad_bot = self.img_size[0] - x_scaled.shape[-2] - pad_top
            x_padded = nn.functional.pad(input=x_scaled, pad=(0, 0, pad_top, pad_bot))
        elif target_ratio < actual_ratio:
            x_scaled = nn.functional.interpolate(x, (self.img_size[0], round(self.img_size[0] / actual_ratio)))
            pad_left = (self.img_size[1] - x_scaled.shape[-1]) // 2
            pad_right = self.img_size[1] - x_scaled.shape[-1] - pad_left
            x_padded = nn.functional.pad(input=x_scaled, pad=(pad_left, pad_right))
        else:
            x_padded = x

        return x_padded, (pad_top, pad_bot, pad_left, pad_right)

    def depad(self, x, target_size, pads):
        x_depad = x[..., pads[0]: -pads[1] if pads[1] else None, pads[2]: -pads[3] if pads[3] else None]
        x_scaled = nn.functional.interpolate(x_depad, target_size)
        return x_scaled

    def forward(self, x):
        # x = torch.cat([x, x], dim=0)
        while len(x.shape) < 4:
            x = x[None, :]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        x = normalize(x)
        original_size = x.shape[-2:]
        x, pads = self.resize(x)
        feat = self.resnet(x)
        out = self.final_stage(feat)

        bps = self.generate_final_bps(out, x)
        bps = self.depad(bps, original_size, pads)

        return bps, [out, ]

    def freeze_encoder(self, freeze=True):
        self.resnet.freeze(freeze=freeze)
        logger.info("Encoder frozen")

    def freeze_deconv(self, freeze=True):
        self.final_stage.freeze(freeze=freeze)
        logger.info("Deconv frozen")

    def load_state_dict(self, state_dict, strict=True):
        state_dict = OrderedDict({k.replace('module.', ''): v
                                  for k, v in state_dict.items()})

        nn.Module.load_state_dict(self, state_dict=state_dict, strict=True)

    def generate_final_bps(self, out_coco, img, sigma=12):
        img_size = img.shape[-2:]

        aps_in_coco = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        out = out_coco[:, aps_in_coco]

        v, kps = out.view(*out.shape[:-2], -1).max(dim=-1)
        kps = torch.stack((kps.div(out.shape[-1], rounding_mode='trunc'), kps % out.shape[-1]), -1)

        # Add thorax
        thorax_kp = torch.round((kps[:, 1, :] + kps[:, 4, :]) / 2.)
        kps = torch.cat((kps[:, :1], thorax_kp[:, None], kps[:, 1:]), dim=1)

        ratios = torch.tensor([x / y for x, y in zip(img.shape[-2:], out.shape[-2:])], dtype=kps.dtype,
                              device=kps.device)
        kps = torch.round(kps * ratios[None, None, :])

        thorax_v = (v[:, 1] + v[:, 4]) / 2.
        v = torch.cat((v[:, :1], thorax_v[:, None], v[:, 1:]), dim=1)
        # v = v > 0.25
        v = v > 0.15

        result = torch.zeros(kps.shape[:-1] + img_size, dtype=out.dtype, device=out.device)
        if self.mesh_grid is None:
            self.mesh_grid = [t.to(out.device) for t in
                              torch.meshgrid(torch.arange(img_size[0]), torch.arange(img_size[1]), indexing="ij")]
        yy, xx = self.mesh_grid

        for b, kp in enumerate(kps):
            for k, point in enumerate(kp):
                if not v[b][k]:
                    pass
                result[b, k] = torch.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))

        return result


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(gen_final=True):
    num_layers = 50
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, gen_final=gen_final)

    # model.init_weights('assets/autob_vis2_soft_fine2.pth.tar')
    model.init_weights('assets/coco_vis2_0_novis.pth.tar')
    for p in model.parameters():
        p.requires_grad = False
    if torch.cuda.is_available():
        model.cuda()
    return model
