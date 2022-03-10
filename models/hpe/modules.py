import os
import logging

import torch
import torch.nn as nn
import glob
from collections import OrderedDict
from .resnet import Bottleneck, BN_MOMENTUM

logger = logging.getLogger(__name__)


class DeconvStage(nn.Module):
    def __init__(self, inplanes):
        super(DeconvStage, self).__init__()

        self.inplanes = inplanes
        self.deconv_with_bias = False
        self.nb_joints = 17

        num_layers = 3
        self.deconv_layers = self._make_deconv_layer(num_layers, [256] * 3, [4] * 3)

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=self.nb_joints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, feats):
        x = self.deconv_layers(feats)
        x = self.final_layer(x)

        return x

    def freeze(self, freeze=True):
        for layer in [
            self.deconv_layers,
            self.final_layer,
        ]:
            for p in layer.parameters():
                p.requires_grad = not freeze


class DeconvStageVis(DeconvStage):
    def __init__(self, inplanes, block, cfg):
        super(DeconvStageVis, self).__init__(inplanes)

        self.nb_vis = cfg.MODEL.NB_VIS
        extra = cfg.MODEL.EXTRA

        # New visibility deductor
        self.fc = self._make_fc(512 * block.expansion,
                                extra.HEATMAP_SIZE,
                                extra.NUM_DECONV_LAYERS,
                                fc_sizes=cfg.MODEL.EXTRA.NUM_LINEAR_LAYERS
                                )

    def _make_fc(self, input_channel=512 * 4, hm_size=[64, 48], deconv_ratio=3, fc_sizes=[4096, 2048, 1024]):
        out_conv = 512
        max_pool = 2
        layers = [
            Bottleneck(input_channel, int(out_conv / Bottleneck.expansion),
                       downsample=nn.Conv2d(input_channel, out_conv, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(out_conv, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=max_pool, stride=max_pool, padding=0), nn.Flatten(), ]

        # input layer
        layers += [
            nn.Linear(int(out_conv * hm_size[0] * hm_size[1] / (max_pool * 2 ** deconv_ratio) ** 2), fc_sizes[0]),
            nn.ReLU()]

        # hidden layers
        for i in range(len(fc_sizes) - 1):
            layers.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))
            layers.append(nn.ReLU())

        # output layers
        layers.append(nn.Linear(fc_sizes[-1], self.nb_vis * self.nb_joints))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, feats):
        x = self.deconv_layers(feats)
        x = self.final_layer(x)

        vis_feats = self.fc(feats)
        vis_preds = vis_feats.reshape((vis_feats.shape[0], self.nb_joints, self.nb_vis))

        return x, vis_preds


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()

        self.final_stage = None

    def forward(self, x):
        raise NotImplementedError()

    def init_weights(self, pretrained_pth=''):
        if os.path.isfile(pretrained_pth):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.final_stage.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.final_stage.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_stage.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained_pth))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained_pth)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained_pth))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error(f'=> imagenet pretrained model dose not exist : {pretrained_pth}')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')
