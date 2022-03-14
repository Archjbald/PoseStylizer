from collections import OrderedDict

import torch

from .PoseCycleHPENet import TransferCycleHPEModel
from .base_model import BaseModel
from . import networks

from util.image_pool import ImagePool
from .hpe.simple_bl import get_pose_net
from .PoseCycleHPENet import TransferCycleHPEModel


class TransferCycleHPEModelD(TransferCycleHPEModel, BaseModel):
    def name(self):
        return 'TransferCycleHPEModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt

        self.model_names = ['netG', 'netHPE']

        # define networks (both generator and discriminator)
        input_nc = [opt.P_input_nc, opt.BP_input_nc, opt.BP_input_nc]

        self.netG = networks.define_G(input_nc, opt.P_input_nc, opt.ngf, opt.which_model_netG, opt.norm,
                                      not opt.no_dropout, opt.use_transfer_layer, opt.init_type,
                                      self.gpu_ids, n_downsampling=opt.G_n_downsampling, opt=opt)

        self.netHPE = get_pose_net()
        self.lambda_HPE = opt.lambda_HPE

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.P_input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          not opt.no_dropout_D,
                                          n_downsampling=opt.D_n_downsampling)
            self.model_names.append('netD')

        which_epoch = opt.which_epoch
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_PB, 'netD', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterion_GAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_idt = torch.nn.L1Loss()
            self.criterion_HPE = torch.nn.MSELoss()

            # lambdas:
            lambdas = ['GAN', 'cycle', 'identity']
            for lbd in lambdas:
                lbd = f'lambda_{lbd}'
                setattr(self, lbd, getattr(opt, lbd))

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')

    def test(self):
        with torch.no_grad():
            self.forward()

    # D: take(P, B) as input
    def backward_D(self, backward=True):
        nbj = self.opt.BP_input_nc
        loss_D = 0
        pairs = [(self.input_P2, self.fake_P2), (self.input_P1, self.fake_P1)]
        for pair in pairs:
            real = pair[0]
            fake = self.fake_pool.query(pair[1].data)
            pred_real = self.netD(real)
            loss_real = self.criterion_GAN(pred_real, True) * self.lambda_GAN
            # Fake
            pred_fake = self.netD(fake.detach())
            loss_fake = self.criterion_GAN(pred_fake, False) * self.lambda_GAN
            # Combined loss
            loss = (loss_real + loss_fake) * 0.5
            # backward
            if backward:
                loss.backward()
            loss_D += loss
        self.loss_D = loss_D.item() / len(pairs)

    def backward_G(self, backward=True):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_cycle = self.lambda_cycle
        lambda_HPE = self.lambda_HPE

        # Adversarial loss
        self.loss_G_1 = 0
        self.loss_G_2 = 0

        self.loss_G_GAN_1 = self.criterion_GAN(self.netD(self.fake_P2), True)
        self.loss_G_1 += self.loss_G_GAN_1
        self.loss_G_GAN_2 = self.criterion_GAN(self.netD(self.fake_P1), True)
        self.loss_G_2 += self.loss_G_GAN_2

        # Fake_qualities
        quality_1 = 1. - self.loss_G_1.item()
        quality_2 = 1. - self.loss_G_2.item()

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_1 = self.netG([self.input_P1, self.input_BP1, self.input_BP1])
            self.loss_idt_1 = self.criterion_idt(self.idt_1, self.input_P1) * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_2 = self.netG([self.input_P2, self.input_BP2, self.input_BP2])
            self.loss_idt_2 = self.criterion_idt(self.idt_2, self.input_P2) * lambda_idt
        else:
            self.loss_idt_1 = 0
            self.loss_idt_2 = 0

        # HPE Loss
        self.loss_HPE_1 = 0
        self.loss_HPE_2 = 0
        self.loss_HPE_1_cycle = 0
        self.loss_HPE_2_cycle = 0
        if lambda_HPE:
            self.loss_HPE_1 = self.criterion_HPE(self.fake_BP1, self.input_BP1) * lambda_HPE
            self.loss_HPE_2 = self.criterion_HPE(self.fake_BP2, self.input_BP2) * lambda_HPE

            with torch.no_grad():
                self.loss_HPE_1_cycle = self.criterion_HPE(self.netHPE(self.rec_P1),
                                                           self.input_BP1) * lambda_HPE * quality_1
                self.loss_HPE_2_cycle = self.criterion_HPE(self.netHPE(self.rec_P2),
                                                           self.input_BP2) * lambda_HPE * quality_2

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_1 = self.criterion_cycle(self.rec_P1, self.input_P1) * lambda_cycle
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_2 = self.criterion_cycle(self.rec_P2, self.input_P2) * lambda_cycle

        # combined loss and calculate gradients
        self.loss_G = (self.loss_G_1 + self.loss_G_1 + self.loss_cycle_1 + self.loss_cycle_2 +
                       self.loss_idt_1 + self.loss_idt_2 + self.loss_HPE_1 + self.loss_HPE_2 +
                       self.loss_HPE_1_cycle + self.loss_HPE_2_cycle)
        if backward:
            self.loss_G.backward()

    def optimize_parameters(self, backward=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(backward=backward)  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        self.set_requires_grad([self.netD], True)
        for i in range(self.opt.DG_ratio):
            self.optimizer_D.zero_grad()
            self.backward_D(backward=backward)
            self.optimizer_D.step()

    def save(self, label, epoch, total_steps):
        self.save_network(self.netG, 'netG', label, self.gpu_ids, epoch, total_steps)
        self.save_network(self.netD, 'netD', label, self.gpu_ids, epoch, total_steps)

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['D'] = self.loss_D
        ret_errors['G_1'] = self.loss_G_1.item()
        ret_errors['G_2'] = self.loss_G_2.item()
        ret_errors['cycle_1'] = self.loss_cycle_1.item()
        ret_errors['cycle_2'] = self.loss_cycle_2.item()
        ret_errors['idt_1'] = self.loss_idt_1.item()
        ret_errors['idt_2'] = self.loss_idt_2.item()
        if self.lambda_HPE:
            ret_errors['HPE_1'] = self.loss_HPE_1.item()
            ret_errors['HPE_2'] = self.loss_HPE_2.item()

        return ret_errors
