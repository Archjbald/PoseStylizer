import numpy as np
import torch
from collections import OrderedDict

from .base_model import BaseModel
from util.image_pool import ImagePool
from . import networks
import util.util as util
from losses.L1_plus_perceptualLoss import PerceptualLoss
from losses.color_loss import ColorLoss


class TransferCycleModel(BaseModel):
    """
    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def name(self):
        return 'TransferCycleModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt

        self.model_names = ['netG']

        # define networks (both generator and discriminator)
        input_nc = [opt.P_input_nc, opt.BP_input_nc, opt.BP_input_nc]

        self.netG = networks.define_G(input_nc, opt.P_input_nc, opt.ngf, opt.which_model_netG, opt.norm,
                                      not opt.no_dropout, opt.use_transfer_layer, opt.init_type,
                                      self.gpu_ids, n_downsampling=opt.G_n_downsampling, opt=opt)

        if self.isTrain:
            self.netD_list = []
            use_sigmoid = opt.no_lsgan
            if opt.with_D_simple:
                self.netD = networks.define_D(opt.P_input_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                              not opt.no_dropout_D,
                                              n_downsampling=opt.D_n_downsampling)
                self.model_names.append('netD')
                self.netD_list.append(self.netD)
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc + opt.BP_input_nc, opt.ndf,
                                                 opt.which_model_netD,
                                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                                 not opt.no_dropout_D,
                                                 n_downsampling=opt.D_n_downsampling)
                self.model_names.append('netD_PB')
                self.netD_list.append(self.netD_PB)
            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc + opt.P_input_nc, opt.ndf,
                                                 opt.which_model_netD,
                                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                                 not opt.no_dropout_D,
                                                 n_downsampling=opt.D_n_downsampling)
                self.model_names.append('netD_PP')
                self.netD_list.append(self.netD_PP)

        which_epoch = opt.which_epoch
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_simple:
                    self.load_network(self.netD, 'netD', which_epoch)
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)

        self.use_mask = self.opt.use_mask if self.isTrain else False
        if self.use_mask:
            self.get_mask = util.box_from_pose

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterion_GAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_idt = torch.nn.L1Loss()

            if opt.L1_type == 'l1_plus_perL1':
                self.criterion_cycle = PerceptualLoss(1., opt.perceptual_layers, self.gpu_ids)
            else:
                self.criterion_cycle = torch.nn.L1Loss()

            # lambdas:
            lambdas = ['GAN', 'cycle', 'identity', 'adversarial', 'patch']
            for lbd in lambdas:
                lbd = f'lambda_{lbd}'
                setattr(self, lbd, getattr(opt, lbd))

            if self.lambda_patch > 0:
                self.criterion_patch = ColorLoss(opt)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_simple:
                self.fake_pool = ImagePool(opt.pool_size)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr,
                                                    betas=(opt.beta1, 0.999))
            if opt.with_D_PB:
                self.fake_PB_pool = ImagePool(opt.pool_size)
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr,
                                                       betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.fake_PP_pool = ImagePool(opt.pool_size)
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr,
                                                       betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_simple:
                self.optimizers.append(self.optimizer_D)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')

    def set_input(self, input):
        self.input_P1, self.input_BP1 = input['P1'], input['BP1'][:, :18]
        self.input_P2, self.input_BP2 = input['P2'], input['BP2'][:, :18]
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]

        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()

        if self.use_mask:
            self.mask_1 = self.get_mask(self.input_BP1)
            self.input_P1 *= self.mask_1
            self.mask_2 = self.get_mask(self.input_BP2)
            self.input_P2 *= self.mask_2

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_P2 = self.netG([self.input_P1, self.input_BP1, self.input_BP2])  # G_A(A)
        if self.use_mask:
            self.fake_P2 *= self.get_mask(self.input_BP2)
        self.rec_P1 = self.netG([self.fake_P2, self.input_BP2, self.input_BP1])  # G_B(G_A(A))
        if self.use_mask:
            self.rec_P1 *= self.get_mask(self.input_BP1)
        self.fake_P1 = self.netG([self.input_P2, self.input_BP2, self.input_BP1])  # G_B(B)
        if self.use_mask:
            self.fake_P1 *= self.get_mask(self.input_BP1)
        self.rec_P2 = self.netG([self.fake_P1, self.input_BP1, self.input_BP2])  # G_A(G_B(B))
        if self.use_mask:
            self.rec_P2 *= self.get_mask(self.input_BP2)

        if not self.isTrain or self.lambda_identity:
            self.idt_P1 = self.netG([self.input_P1, self.input_BP1, self.input_BP1])
            self.idt_P2 = self.netG([self.input_P2, self.input_BP2, self.input_BP2])

    def backward_D_basic(self, netD, real, fake, backward=True):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN(pred_real, True) * self.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False) * self.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        if backward:
            loss_D.backward()

        return loss_D

    def backward_D(self, backward=True):
        loss_D = 0
        pairs = [(self.input_P2, self.fake_P2), (self.input_P1, self.fake_P1)]
        for pair in pairs:
            real = pair[0]
            fake = self.fake_pool.query(pair[1].data)
            loss_D += self.backward_D_basic(self.netD, real, fake, backward=backward)
        self.loss_D = loss_D.item()

    # D: take(P, B) as input
    def backward_D_PB(self, backward=True):
        nbj = self.opt.BP_input_nc
        loss_D_PB = 0
        pairs = [(self.input_P2, self.input_BP2, self.fake_P2), (self.input_P1, self.input_BP1, self.fake_P1)]
        for pair in pairs:
            real_PB = torch.cat((pair[0], pair[1][:, :nbj]), 1)
            fake_PB = self.fake_PB_pool.query(torch.cat((pair[2], pair[1][:, :nbj]), 1).data)
            loss_D_PB += self.backward_D_basic(self.netD_PB, real_PB, fake_PB, backward=backward)
        self.loss_D_PB = loss_D_PB.item() / len(pairs)

    # D: take(P, P') as input
    def backward_D_PP(self, backward=True):
        loss_D_PP = 0
        pairs = [(self.input_P2, self.input_P1, self.fake_P2), (self.input_P1, self.input_P2, self.fake_P1)]
        for pair in pairs:
            real_PP = torch.cat((pair[0], pair[1]), 1)
            fake_PP = self.fake_PP_pool.query(torch.cat((pair[2], pair[1]), 1).data)
            loss_D_PP += self.backward_D_basic(self.netD_PP, real_PP, fake_PP, backward=backward)
        self.loss_D_PP = loss_D_PP.item()

    def backward_G(self, backward=True):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_cycle = self.lambda_cycle
        lambda_adv = self.lambda_adversarial
        lambda_patch = self.lambda_patch

        # Adversarial loss
        self.loss_adv = 0.
        loss_adv_1 = 0.
        loss_adv_2 = 0.
        if self.opt.with_D_simple:
            loss_adv_1 += self.criterion_GAN(self.netD(self.fake_P2), True)
            loss_adv_2 += self.criterion_GAN(self.netD(self.fake_P1), True)

        if self.opt.with_D_PB:
            loss_adv_1 += self.criterion_GAN(self.netD_PB(torch.cat((self.fake_P2, self.input_BP2), 1)), True)
            loss_adv_2 += self.criterion_GAN(self.netD_PB(torch.cat((self.fake_P1, self.input_BP1), 1)), True)

        if self.opt.with_D_PP:
            loss_adv_1 += self.criterion_GAN(self.netD_PP(torch.cat((self.fake_P2, self.input_P1), 1)), True)
            loss_adv_2 += self.criterion_GAN(self.netD_PP(torch.cat((self.fake_P1, self.input_P2), 1)), True)

        self.loss_adv = (loss_adv_1 + loss_adv_2) / 2. * lambda_adv

        # Identity loss
        self.loss_idt = 0.
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.loss_idt += self.criterion_idt(self.idt_P1, self.input_P1) * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.loss_idt += self.criterion_idt(self.idt_P2, self.input_P2) * lambda_idt
            self.loss_idt /= 2.

        # Patch loss
        self.loss_patch = 0.
        if lambda_patch > 0:
            self.loss_patch += self.criterion_patch(self.input_P1, self.input_BP1, self.fake_P2, self.input_BP2)
            self.loss_patch += self.criterion_patch(self.input_P2, self.input_BP2, self.fake_P1, self.input_BP1)

        self.loss_cycle = 0.
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle += self.criterion_cycle(self.rec_P1, self.input_P1) * lambda_cycle
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle += self.criterion_cycle(self.rec_P2, self.input_P2) * lambda_cycle
        self.loss_cycle /= 2.

        # combined loss and calculate gradients
        self.loss_G = (self.loss_adv + self.loss_cycle + self.loss_idt + self.loss_patch)
        if backward:
            self.loss_G.backward()

    def optimize_parameters(self, backward=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G  # Ds require no gradients when optimizing Gs
        self.set_requires_grad(self.netD_list, False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(backward=backward)  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        self.set_requires_grad(self.netD_list, True)
        # D
        if self.opt.with_D_simple:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D.zero_grad()
                self.backward_D(backward=backward)
                self.optimizer_D.step()

        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP(backward=backward)
                self.optimizer_D_PP.step()

        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB(backward=backward)
                self.optimizer_D_PB.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.opt.with_D_simple:
            ret_errors['D'] = self.loss_D
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB

        ret_errors['adv'] = self.loss_adv.item()
        ret_errors['cycle'] = self.loss_cycle.item()
        ret_errors['idt'] = self.loss_idt.item()
        if self.loss_patch > 0:
            ret_errors['patch'] = self.loss_patch.item()
        return ret_errors

    def get_current_p2(self):
        return util.tensor2im(self.fake_P2.data)

    def get_current_visuals(self):
        nbj = self.opt.BP_input_nc
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)
        fake_P2 = util.tensor2im(self.fake_P2.data)
        rec_P1 = util.tensor2im(self.rec_P1.data)
        idt_P2 = util.tensor2im(self.idt_P2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1[:, :nbj].data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2[:, :nbj].data)[0]

        fake_BP2 = util.draw_pose_from_map(self.fake_BP2.data)[0]

        imgs = [input_P1, input_BP1, input_P2, input_BP2, fake_P2, fake_BP2, rec_P1, idt_P2]
        vis = np.zeros((height, width * len(imgs), 3)).astype(np.uint8)  # h, w, c
        for i, img in enumerate(imgs):
            vis[:, width * i:width * (i + 1), :] = img

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label, epoch, total_steps):
        self.save_network(self.netG, 'netG', label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_simple:
            self.save_network(self.netD, 'netD', label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB, 'netD_PB', label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids, epoch, total_steps)

    def cleanse(self):
        # output
        del self.fake_P1, self.fake_P2, self.rec_P1, self.rec_P2

        # loss G
        del self.idt_P1, self.loss_idt_1, self.idt_P2, self.loss_idt_2
        del self.loss_G_GAN_PB_1, self.loss_G_GAN_PP_1, self.loss_G_1
        del self.loss_G_GAN_PB_2, self.loss_G_GAN_PP_2, self.loss_G_2
        del self.loss_cycle_1, self.loss_cycle_2, self.loss_G

        # loss D
        del self.loss_D_PB, self.loss_D_PP

        torch.cuda.empty_cache()
