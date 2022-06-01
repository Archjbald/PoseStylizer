import torch

from .PoseCycleNet import TransferCycleModel
from .hpe.openpose import get_pose_net
import util.util as util


class TransferCycleHPEModel(TransferCycleModel):
    def name(self):
        return 'PATNCycle'

    def initialize(self, opt):
        TransferCycleModel.initialize(self, opt)

        self.model_names.append('netHPE')
        self.netHPE = get_pose_net()

        if self.isTrain:
            self.lambda_HPE = opt.lambda_HPE
            self.criterion_HPE = torch.nn.MSELoss()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_P2 = self.netG([self.input_P1, self.input_BP1, self.input_BP2])  # G_A(A)
        self.fake_BP2 = self.netHPE(self.fake_P2)
        if self.use_mask:
            self.fake_P2 *= self.mask_2
        self.rec_P1 = self.netG([self.fake_P2, self.fake_BP2 if self.opt.fake_bp_cycle else self.input_BP2,
                                 self.input_BP1])  # G_B(G_A(A))
        if self.use_mask:
            self.rec_P1 *= self.get_mask(self.input_BP1)

        self.fake_P1 = self.netG([self.input_P2, self.input_BP2, self.input_BP1])  # G_B(B)
        self.fake_BP1 = self.netHPE(self.fake_P1)
        if self.use_mask:
            self.fake_P1 *= self.mask_1
        self.rec_P2 = self.netG([self.fake_P1, self.fake_BP1 if self.opt.fake_bp_cycle else self.input_BP1,
                                 self.input_BP2])  # G_A(G_B(B))
        if self.use_mask:
            self.rec_P2 *= self.get_mask(self.input_BP2)

        if not self.isTrain or self.lambda_identity:
            self.idt_P1 = self.netG([self.input_P1, self.input_BP1, self.input_BP1])
            self.idt_P2 = self.netG([self.input_P2, self.input_BP2, self.input_BP2])

        self.fake_BP1 = self.netHPE(self.fake_P1)
        self.fake_BP2 = self.netHPE(self.fake_P2)

    def evaluate_HPE(self, fake, real):
        annotated = real.view(*real.shape[:-2], -1).max(dim=-1)[0] > 0
        loss = self.criterion_HPE(fake * annotated[:, :, None, None], real) * self.lambda_HPE
        return loss

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

        # HPE Loss
        self.loss_HPE = 0.
        if self.lambda_HPE:
            self.loss_HPE += self.evaluate_HPE(self.fake_BP1, self.input_BP1)
            self.loss_HPE += self.evaluate_HPE(self.fake_BP2, self.input_BP2)
            self.loss_HPE /= 2.

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
        self.loss_G = self.loss_adv + self.loss_cycle + self.loss_idt + self.loss_HPE +self.loss_patch
        if backward:
            self.loss_G.backward()

    def get_current_errors(self):
        ret_errors = TransferCycleModel.get_current_errors(self)
        if self.lambda_HPE:
            ret_errors['HPE'] = self.loss_HPE.item()

        return ret_errors
