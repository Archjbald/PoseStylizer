import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as fn

from util.util import get_kps

import sys


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #     print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'switchable':
        from .model_utils import SwitchNorm2d
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, use_transfer_layer=False,
             init_type='normal', gpu_ids=[], n_downsampling=2, opt=None):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG in ['APS']:
        from models.APS import stylegenerator
        netG = stylegenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              use_transfer_layer=use_transfer_layer, n_blocks=9, gpu_ids=gpu_ids,
                              n_downsampling=n_downsampling, opt=opt)
    elif which_model_netG in ['PATN']:
        from models.PATN import stylegenerator
        input_nc = [opt.P_input_nc, opt.BP_input_nc + opt.BP_input_nc]
        netG = stylegenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                              gpu_ids=gpu_ids, n_downsampling=n_downsampling)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    """
    if len(gpu_ids) > 1:
        netG = nn.DataParallel(netG, device_ids=gpu_ids)
    """

    netG.cuda()
    init_weights(netG, init_type=init_type)
    return netG


def define_F(netF, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    net = None
    if netF == 'sample':
        net = PatchSamplePoseF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSamplePoseF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)

    """
    if len(gpu_ids) > 1:
        net = nn.DataParallel(net, device_ids=gpu_ids)
    """

    net.cuda()
    return net


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], use_dropout=False,
             n_downsampling=2, linear_size=None):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netD == 'resnet':
        netD = ResnetDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_layers_D,
                                   gpu_ids=[], padding_type='reflect', use_sigmoid=use_sigmoid,
                                   n_downsampling=n_downsampling, linear_size=linear_size)
    elif which_model_netD == 'nlayer':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    """
    if len(gpu_ids) > 1:
        netD = nn.DataParallel(netD, device_ids=gpu_ids)
    #     if use_gpu:
    #         netD.cuda(gpu_ids[0])
    """

    netD.cuda()
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################    


class WassersteinLoss:
    def __call__(self, input, target_is_real):
        return torch.mean(input) * (-1 if not target_is_real else 1)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, use_wgan=False):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.use_target = True
        if use_wgan:
            self.use_target = False
            self.loss = WassersteinLoss()
        elif use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if self.use_target:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)
        else:
            return self.loss(input, target_is_real)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetDiscriminator(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
                 padding_type='reflect', use_sigmoid=False, n_downsampling=2, linear_size=None):
        assert (n_blocks >= 0)
        super(ResnetDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_sigmoid = use_sigmoid

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.2, True)]

        # n_downsampling = 2
        if n_downsampling <= 2:
            for i in range(n_downsampling):
                mult = 2 ** i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.LeakyReLU(0.2, True)]
        elif n_downsampling == 3:
            mult = 2 ** 0
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(0.2, True)]
            mult = 2 ** 1
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(0.2, True)]
            mult = 2 ** 2
            model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult),
                      nn.LeakyReLU(0.2, True)]

        if n_downsampling <= 2:
            mult = 2 ** n_downsampling
        else:
            mult = 4
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]
        if linear_size:
            self.use_linear = True
            self.linear = nn.Linear(linear_size[0] * linear_size[1] * ngf // mult, 1)
        else:
            self.use_linear = False

        self.model = nn.Sequential(*model)

    def forward(self, input, mask=None, use_sigmoid=None):
        if use_sigmoid is None:
            use_sigmoid = self.use_sigmoid
        y = self.model(input)
        if use_sigmoid:
            y = torch.sigmoid(y)
        if mask is not None:
            mask = F.interpolate(mask, size=(y.shape[2], y.shape[3]), align_corners=False)
            y = y * mask
        if self.use_linear:
            y = y.view(-1, self.linear.in_features)
            y = self.linear(y)
        return y


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = (4, 4)
        pad = (1, 1)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=(2, 2), padding=pad),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=(2, 2), padding=pad, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=(1, 1), padding=pad, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=(1, 1), padding=pad)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.to(self.gpu_ids[0])
        init_weights(self, self.init_type)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    # patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchSamplePoseF(PatchSampleF):
    @staticmethod
    def get_ids_kps(bp1, bp2, scales, patch_sizes, num_patches=64, in_mask=False):
        """
        Generate list of patch ids according to skeleton.
        Skeleton ids are shared across scales, random are not
        :param bp1: skeleton 1 (BxCxHxW)
        :param bp2: skeleton 2 (BxCxHxW)
        :param scales: shapes of each feature vector (FxBxCxHxW)
        :param patch_sizes: sizes of patchs to extract
        :param num_patches: number of expected patches
        :return: list of ids per scales, num_patches
        """
        device = bp1.device
        if len(patch_sizes) == 1:
            patch_sizes = patch_sizes * len(scales)

        assert len(patch_sizes) == len(scales), 'Wrong number of patch sizes'

        patch_shapes = [(size, size) for size in patch_sizes]
        patch_sizes = [shape[0] * shape[1] for shape in patch_shapes]
        # bp1 = torch.cat((bp1, bp1))
        # bp2 = torch.cat((bp2, bp2))
        assert bp1.shape == bp2.shape
        B, C, H, W = bp1.shape

        if isinstance(num_patches, int):
            num_patches = [max(size * C, num_patches) for size in patch_sizes]

        kps_1, v_1 = get_kps(bp1, w=W)
        kps_2, v_2 = get_kps(bp2, w=W)

        v_12 = (v_1 * v_2).nonzero()
        ratios = scales / torch.tensor([H, W], device=device)

        patch_ids = []
        for s, scale in enumerate(scales):
            patch_id = -torch.ones((2, B, num_patches[s]), dtype=torch.long, device=device)

            # Keypoint patches
            h, w = scale.tolist()
            idx = torch.arange(0, h * w).view(h, w)
            pad_x = patch_shapes[s][0] // 2
            pad_y = patch_shapes[s][1] // 2
            idx_pad = F.pad(idx, (pad_y, pad_y, pad_x, pad_x), value=-1)
            idx_patches = idx_pad.unfold(0, patch_shapes[s][0], 1).unfold(1, patch_shapes[s][1], 1).contiguous().view(h,
                                                                                                                      w,
                                                                                                                      -1)

            ratio = ratios[s]
            for i, kps in enumerate([kps_1, kps_2]):
                kp = kps[v_12[:, 0], v_12[:, 1]]
                kp = (kp * ratio[None, :]).round().type(torch.int)
                for d, dd in enumerate((h, w)):
                    kp[:, d][kp[:, d] >= dd] = dd - 1
                coords = torch.stack([v_12[:, 0], v_12[:, 1], kp[:, 0], kp[:, 1]], dim=1)
                for coord in coords:
                    patch_id[i, coord[0], coord[1] * patch_sizes[s]:(coord[1] + 1) * patch_sizes[s]] = idx_patches[
                        coord[-2], coord[-1]]
            patch_ids.append(patch_id)

        # Random Patches
        if not in_mask:
            mask_1 = bp1.sum(dim=1) > 0.1
            mask_2 = bp2.sum(dim=1) > 0.1
            mask_12 = mask_1 * mask_2
        else:
            mask_12 = torch.zeros_like(bp1[:, 0]).type(torch.bool)

        for s, scale in enumerate(scales):
            # idx_random = (patch_ids[s] == -1).nonzero()  # [2, B, num_patch]
            # nb_random = [len(idx_random[idx_random[:, 0] == i]) for i in range(2)]  # [2, nz]

            idx_random = [[patch_ids[s][i, j] == -1 for j in range(B)] for i in range(2)]

            mask_12_scale = fn.resize(mask_12, scale.tolist())
            # mask_12_idx = (~mask_12_scale).nonzero()  # [B, H, W]
            mask_12_flat = mask_12_scale.view(mask_12_scale.shape[0], -1)  # [B, HW]

            random_ids = torch.tensor(np.random.permutation((scale[0] * scale[1]).item()),
                                      dtype=torch.long, device=device)
            for im in range(B):
                mask = ~mask_12_flat[im]
                for i in range(2):
                    idxs = idx_random[i][im]
                    try:
                        patch_ids[s][i, im, idxs] = random_ids[mask][:len(idxs)][idxs]
                    except IndexError as err:
                        print('***', bp1.shape, s, scales, patch_ids[s].shape, random_ids.shape, mask.sum(),
                              random_ids[mask].shape, len(idxs))
                        raise err

        patch_ids = [[pi[i] for pi in patch_ids] for i in range(2)]
        return patch_ids, num_patches

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            num_patch = num_patches[feat_id]
            if num_patch > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    raise ValueError('Expecting Patch ids from BPs')
                # patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = torch.zeros((B, num_patch, C), dtype=feat_reshape.dtype, device=feat_reshape.device)
                for b, pid in enumerate(patch_id):
                    x_sample[b] = feat_reshape[b, pid, :]
                x_sample = x_sample.flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patch == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids
