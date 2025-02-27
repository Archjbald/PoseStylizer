import sys

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os


class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                         channels * 2,
                         gain=1.0,
                         use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class AdaIN(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale=False,
                 use_noise=False,
                 use_pixel_norm=False,
                 use_instance_norm=True,
                 use_styles=True):
        super(AdaIN, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
        else:
            self.noise = None
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None, noise=None):
        if self.noise is not None:
            x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        return x


class PatchNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 factor,
                 dataset=None,
                 res=0,
                 cut=False,
                 bias=True):
        super(PatchNorm, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.instance_norm = InstanceNorm()
        self.pixel_norm = PixelNorm()
        if bias:
            self.fc = nn.Conv2d(in_channels, out_channels * 2, 1, stride=1, padding=0, bias=True)
        else:
            self.fc = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=True)
        self.factor = factor
        self.dataset = dataset
        self.bias = bias

    def forward(self, x, style, norm='InstanceNorm', return_stats=False):
        x = self.act(x)
        if self.bias:
            beta, gamma = self.fc(style).chunk(2, 1)

            if return_stats:
                beta_untiled, gamma_untiled = beta, gamma
            beta, gamma = tile(beta, dim=2, n_tile=self.factor), tile(gamma, dim=2, n_tile=self.factor)
            beta, gamma = tile(beta, dim=3, n_tile=self.factor), tile(gamma, dim=3, n_tile=self.factor)
        else:
            gamma = self.fc(style)
            if return_stats:
                beta_untiled, gamma_untiled = None, gamma
            gamma = tile(gamma, dim=2, n_tile=self.factor)
            gamma = tile(gamma, dim=3, n_tile=self.factor)

        if gamma.shape[3] > x.shape[3]:
            tile_cut = (gamma.shape[3] - x.shape[3]) // 2
            if self.bias:
                beta, gamma = (beta[:, :, :, tile_cut:x.shape[3] + tile_cut],
                               gamma[:, :, :, tile_cut:x.shape[3] + tile_cut])
            else:
                gamma = gamma[:, :, :, tile_cut:x.shape[3] + tile_cut]
        if norm is None:
            x_mean = calc_patch_mean(x, self.factor)
            x = x - x_mean
            tmp = torch.mul(x, x)
            x_std = torch.rsqrt(calc_patch_mean(tmp, self.factor) + 1e-8)
            x = x * x_std
        if norm == 'InstanceNorm':
            x = self.instance_norm(x)
        if norm == 'PixelNorm':
            x = self.pixel_norm(x)
        if self.bias:
            x = x * (gamma + 1.) + beta
        else:
            x = x * (gamma + 1.)
        if return_stats:
            return x, beta_untiled, gamma_untiled
        else:
            return x


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):

        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def calc_patch_mean(x, factor=8):
    h, w = int(x.shape[2] / factor), int(x.shape[3] / factor)
    x_mean = F.adaptive_avg_pool2d(x, (h, w))
    x_mean = tile(x_mean, dim=2, n_tile=factor)
    x_mean = tile(x_mean, dim=3, n_tile=factor)
    return x_mean


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)  # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)  # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class Conv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
