from collections import OrderedDict

import torch

from .PoseCycleNet import TransferCycleModel
from .hpe.simple_bl import get_pose_net


class TransferCycleHPEModel(TransferCycleModel):
    def name(self):
        return 'TransferCycleHPEModel'

    def initialize(self, opt):
        TransferCycleModel.initialize(self, opt)

        self.model_names.append('netHPE')
        self.netHPE = get_pose_net()
        self.lambda_HPE = opt.lambda_HPE

        if self.isTrain:
            self.criterion_HPE = torch.nn.MSELoss()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_P2 = self.netG([self.input_P1, self.input_BP1, self.input_BP2])  # G_A(A)
        with torch.no_grad():
            self.fake_BP2 = self.netHPE(self.fake_P2)
        self.rec_P1 = self.netG([self.fake_P2, self.fake_BP2, self.input_BP1])  # G_B(G_A(A))

        self.fake_P1 = self.netG([self.input_P2, self.input_BP2, self.input_BP1])  # G_B(B)
        with torch.no_grad():
            self.fake_BP1 = self.netHPE(self.fake_P1)
        self.rec_P2 = self.netG([self.fake_P1, self.fake_BP1, self.input_BP2])  # G_A(G_B(B))

    def backward_G(self, backward=True):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_cycle = self.lambda_cycle
        lambda_HPE = self.lambda_HPE

        # Adversarial loss
        self.loss_G_1 = 0
        self.loss_G_2 = 0
        GAN_ratio = self.opt.with_D_PP + self.opt.with_D_PB
        if self.opt.with_D_PB:
            self.loss_G_GAN_PB_1 = self.criterion_GAN(self.netD_PB(torch.cat((self.fake_P2, self.input_BP2), 1)), True)
            self.loss_G_1 += self.loss_G_GAN_PB_1 * GAN_ratio
            self.loss_G_GAN_PB_2 = self.criterion_GAN(self.netD_PB(torch.cat((self.fake_P1, self.input_BP1), 1)), True)
            self.loss_G_2 += self.loss_G_GAN_PB_2 * GAN_ratio

        if self.opt.with_D_PP:
            self.loss_G_GAN_PP_1 = self.criterion_GAN(self.netD_PP(torch.cat((self.fake_P2, self.input_P1), 1)), True)
            self.loss_G_GAN_PP_2 = self.criterion_GAN(self.netD_PP(torch.cat((self.fake_P1, self.input_P2), 1)), True)
            self.loss_G_1 += self.loss_G_GAN_PP_1 * GAN_ratio
            self.loss_G_2 += self.loss_G_GAN_PP_2 * GAN_ratio

        # Fake_qualities
        quality_1 = 1. - self.loss_G_1.item() / GAN_ratio if self.loss_G_1 else 1.
        quality_2 = 1. - self.loss_G_2.item() / GAN_ratio if self.loss_G_2 else 1.

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

    def get_current_errors(self):
        ret_errors = TransferCycleModel.get_current_errors(self)
        if self.lambda_HPE:
            ret_errors['HPE_1'] = self.loss_HPE_1.item()
            ret_errors['HPE_2'] = self.loss_HPE_2.item()

        return ret_errors
