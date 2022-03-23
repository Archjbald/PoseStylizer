from collections import OrderedDict

import torch
from torch.autograd import Variable, grad
from .PoseCycleHPENet import TransferCycleHPEModel
from .base_model import BaseModel
from . import networks

from util.image_pool import ImagePool
from .hpe.simple_bl import get_pose_net
from .PoseCycleHPENet import TransferCycleHPEModel
from losses.L1_plus_perceptualLoss import PerceptualLoss


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
        self.lambda_GAN = 0.2

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
                self.load_network(self.netD, 'netD', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterion_idt = torch.nn.L1Loss()
            self.criterion_HPE = torch.nn.MSELoss()

            if opt.L1_type == 'l1_plus_perL1':
                self.criterion_cycle = PerceptualLoss(1., opt.perceptual_layers, self.gpu_ids)
            else:
                self.criterion_cycle = torch.nn.L1Loss()

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

    # D: take(P, B) as input
    def backward_D(self, backward=True):
        loss_D = 0.
        pairs = [(self.input_P2, self.fake_P2), (self.input_P1, self.fake_P1)]
        for pair in pairs:
            real = pair[0]
            fake = self.fake_pool.query(pair[1].data)
            pred_real = self.netD(real)
            loss_real = - pred_real.mean()
            # Fake
            pred_fake = self.netD(fake.detach())
            loss_fake = pred_fake.mean()

            # Gradient penalty
            eps = Variable(torch.rand(1), requires_grad=True)
            eps = eps.expand(real.size())
            eps = eps.cuda()
            x_tilde = eps * real + (1 - eps) * fake.detach()
            x_tilde = x_tilde.cuda()
            pred_tilde = self.netD(x_tilde)
            gradients = grad(outputs=pred_tilde, inputs=x_tilde,
                             grad_outputs=torch.ones(pred_tilde.size()).cuda(),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

            # Combined loss
            loss = (loss_real + loss_fake) + 10 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss_D += loss

        self.loss_D = loss_D / len(pairs) * self.lambda_GAN
        if backward:
            self.loss_D.backward()

    def backward_G(self, backward=True):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_cycle = self.lambda_cycle

        # Adversarial loss
        self.loss_adv = 0.
        loss_adv_1 = - self.netD(self.fake_P2).mean()
        self.loss_adv += loss_adv_1
        loss_adv_2 = - self.netD(self.fake_P1).mean()
        self.loss_adv += loss_adv_2
        self.loss_adv /= 2.

        # Fake_qualities
        quality_1 = 1. - loss_adv_1.item()
        quality_2 = 1. - loss_adv_2.item()

        # Identity loss
        self.loss_idt = 0.
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_1 = self.netG([self.input_P1, self.input_BP1, self.input_BP1])
            self.loss_idt += self.criterion_idt(self.idt_1, self.input_P1) * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_2 = self.netG([self.input_P2, self.input_BP2, self.input_BP2])
            self.loss_idt += self.criterion_idt(self.idt_2, self.input_P2) * lambda_idt
            self.loss_idt /= 2.

        # HPE Loss
        self.loss_HPE = 0.
        if self.lambda_HPE:
            self.loss_HPE += self.evaluate_HPE(self.fake_BP1, self.input_BP1)
            self.loss_HPE += self.evaluate_HPE(self.fake_BP2, self.input_BP2)
            self.loss_HPE /= 2.

        self.loss_cycle = 0.
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle += self.criterion_cycle(self.rec_P1, self.input_P1) * lambda_cycle
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle += self.criterion_cycle(self.rec_P2, self.input_P2) * lambda_cycle
        self.loss_cycle /= 2.

        # combined loss and calculate gradients
        self.loss_G = self.loss_adv + self.loss_cycle + self.loss_idt + self.loss_HPE
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

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        epoch = scheduler.last_epoch
        progress = min(epoch, self.opt.niter) / self.opt.niter
        self.lambda_percep = 0.2 + progress * 0.6
        self.lambda_GAN = (0.2 + 0.8 * progress) * self.opt.lambda_GAN

    def save(self, label, epoch, total_steps):
        self.save_network(self.netG, 'netG', label, self.gpu_ids, epoch, total_steps)
        self.save_network(self.netD, 'netD', label, self.gpu_ids, epoch, total_steps)

    def get_current_errors(self):
        ret_errors = OrderedDict()
        ret_errors['D'] = self.loss_D
        ret_errors['adv'] = self.loss_adv.item()
        ret_errors['cycle'] = self.loss_cycle.item()
        ret_errors['idt'] = self.loss_idt.item()
        if self.lambda_HPE > 0:
            ret_errors['HPE'] = self.loss_HPE.item()

        return ret_errors
