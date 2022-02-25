import numpy as np
import torch
from collections import OrderedDict

from .base_model import BaseModel
from util.image_pool import ImagePool
from . import networks
import util.util as util


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

        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                      opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                      self.gpu_ids, n_downsampling=opt.G_n_downsampling, opt=opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc + opt.BP_input_nc, opt.ndf,
                                                 opt.which_model_netD,
                                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                                 not opt.no_dropout_D,
                                                 n_downsampling=opt.D_n_downsampling)
                self.model_names.append('netD_PB')

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc + opt.P_input_nc, opt.ndf,
                                                 opt.which_model_netD,
                                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                                 not opt.no_dropout_D,
                                                 n_downsampling=opt.D_n_downsampling)
                self.model_names.append('netD_PP')

        which_epoch = opt.which_epoch
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr,
                                                       betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr,
                                                       betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
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
        if self.opt.dataset_mode in ['keypoint_segmentation']:
            self.input_MP1, self.input_MP2 = input['MP1'], input['MP2']
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]

        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()
            if self.opt.dataset_mode in ['keypoint_segmentation']:
                self.input_MP1 = self.input_MP1.cuda()
                self.input_MP2 = self.input_MP2.cuda()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_P2 = self.netG([self.input_P1, self.input_BP1, self.input_BP2])  # G_A(A)
        self.rec_1 = self.netG([self.fake_P2, self.input_BP2, self.input_BP1])  # G_B(G_A(A))
        self.fake_P1 = self.netG([self.input_P2, self.input_BP2, self.input_BP1])  # G_B(B)
        self.rec_2 = self.netG([self.fake_P1, self.input_BP1, self.input_BP2])  # G_A(G_B(B))

    def test(self):
        with torch.no_grad():
            self.forward()

    def backward_D_basic(self, netD, real, fake, backward=True):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        if backward:
            loss_D.backward()

        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self, backward=True):
        nbj = self.opt.BP_input_nc
        real_PB = torch.cat((self.input_P2, self.input_BP2[:, :nbj]), 1)
        fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_P2, self.input_BP2[:, :nbj]), 1).data)
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB, backward=backward)
        self.loss_D_PB = loss_D_PB.item()

    # D: take(P, P') as input
    def backward_D_PP(self, backward=True):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_P2, self.input_P1), 1).data)
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP, backward=backward)
        self.loss_D_PP = loss_D_PP.item()

    def backward_G(self, backward=True):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_1 = self.netG([self.input_P1, self.input_BP1, self.input_BP1])
            self.loss_idt_1 = self.criterionIdt(self.idt_1, self.input_P1) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_2 = self.netG([self.input_P2, self.input_BP2, self.input_BP2])
            self.loss_idt_2 = self.criterionIdt(self.idt_2, self.input_P2) * lambda_A * lambda_idt
        else:
            self.loss_idt_1 = 0
            self.loss_idt_2 = 0

        self.loss_G_1 = 0
        self.loss_G_2 = 0
        if self.opt.with_D_PB:
            # GAN loss D_A(G_A(A))
            self.loss_G_GAN_PB_1 = self.criterionGAN(self.netD_PB(torch.cat((self.fake_P2, self.input_BP2), 1)), True)
            self.loss_G_1 += self.loss_G_GAN_PB_1

            # GAN loss D_B(G_B(B))
            self.loss_G_GAN_PB_2 = self.criterionGAN(self.netD_PB(torch.cat((self.fake_P1, self.input_BP1), 1)), True)
            self.loss_G_2 += self.loss_G_GAN_PB_2

        if self.opt.with_D_PP:
            self.loss_G_GAN_PP_1 = self.criterionGAN(self.netD_PP(torch.cat((self.fake_P2, self.input_P1), 1)), True)
            self.loss_G_GAN_PP_2 = self.criterionGAN(self.netD_PP(torch.cat((self.fake_P1, self.input_P2), 1)), True)
            self.loss_G_1 += self.loss_G_GAN_PP_1
            self.loss_G_2 += self.loss_G_GAN_PP_2

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_1 = self.criterionCycle(self.rec_1, self.input_P1) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_2 = self.criterionCycle(self.rec_2, self.input_P2) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_1 + self.loss_G_1 + self.loss_cycle_1 + self.loss_cycle_2 + self.loss_idt_1 + self.loss_idt_2
        if backward:
            self.loss_G.backward()

    def optimize_parameters(self, backward=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(backward=backward)  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

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
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB

        ret_errors['G_1'] = self.loss_G_1
        ret_errors['G_2'] = self.loss_G_2
        ret_errors['cycle_1'] = self.loss_cycle_1
        ret_errors['cycle_2'] = self.loss_cycle_2
        return ret_errors

    def get_current_p2(self):
        return util.tensor2im(self.fake_P2.data)

    def get_current_visuals(self):
        nbj = self.opt.BP_input_nc
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1[:, :nbj].data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2[:, :nbj].data)[0]

        fake_P2 = util.tensor2im(self.fake_P2.data)
        vis = np.zeros((height, width * 5, 3)).astype(np.uint8)  # h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width * 2, :] = input_BP1
        vis[:, width * 2:width * 3, :] = input_P2
        vis[:, width * 3:width * 4, :] = input_BP2
        vis[:, width * 4:, :] = fake_P2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def get_current_visuals_widerpose(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map_wider(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map_wider(self.input_BP2.data)[0]

        vis = np.zeros((height, width * 5, 3)).astype(np.uint8)  # h, w, c
        vis[:, :width, :] = input_P1
        vis[:input_BP1.shape[0], width:width + input_BP1.shape[1], :] = input_BP1
        vis[:, width * 2:width * 3, :] = input_P2
        vis[:input_BP2.shape[0], width * 3:width * 3 + +input_BP2.shape[1], :] = input_BP2
        vis[:, width * 4:, :] = 255

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label, epoch, total_steps):
        self.save_network(self.netG, 'netG', label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB, 'netD_PB', label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids, epoch, total_steps)
