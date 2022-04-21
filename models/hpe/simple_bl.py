# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

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

    def switch_gen(self):
        self.gen_final = self.gen_final_train or not self.gen_final

    def forward(self, x):
        # x = torch.cat([x, x], dim=0)
        feat = self.resnet(x)
        out = self.final_stage(feat)

        if self.gen_final:
            out = self.generate_final_bps(out, x)

        return out

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
        v = v > 0.25

        result = torch.zeros(kps.shape[:-1] + img_size, dtype=out.dtype, device=out.device)
        if self.mesh_grid is None:
            self.mesh_grid = [t.to(out.device) for t in
                              torch.meshgrid(torch.arange(img_size[0]), torch.arange(img_size[1]), indexing="ij")]
        yy, xx = self.mesh_grid

        for b, kp in enumerate(kps):
            for k, point in enumerate(kp):
                if not v[b][k]:
                    pass
                print(point.device, xx.device, result.device)
                result[b, k] = torch.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))

        return result


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net():
    num_layers = 50
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers)

    model.init_weights('assets/coco_vis2_0_novis.pth.tar')
    for p in model.parameters():
        p.requires_grad = False
    model.cuda()
    return model
