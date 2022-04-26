import torch

from .PoseCycleNet import TransferCycleModel
from .hpe.simple_bl import get_pose_net
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss


class PATNCycle(TransferCycleModel):
    def name(self):
        return 'PATNCycle'

    def initialize(self, opt):
        opt.with_D_simple = False
        opt.with_D_PP = False
        opt.with_D_PB = True

        opt.use_mask = False

        opt.lambda_identity = 1
        opt.lambda_cycle = 2
        opt.lambda_adversarial = 5
        opt.lambda_HPE = 700
        opt.lambda_patch = 0

        opt.which_model_netG = "PATN"
        opt.norm = 'switchable'

        TransferCycleModel.initialize(self, opt)

        self.model_names.append('netHPE')
        self.gen_final_hpe = False
        self.netHPE = get_pose_net(gen_final=self.gen_final_hpe)

        if self.isTrain:
            self.criterion_cycle = L1_plus_perceptualLoss(1., 0.5, opt.perceptual_layers, self.gpu_ids,
                                                          percep_is_l1=True,
                                                          submodel=self.netHPE.get_feat_extractor())
            self.criterion_idt = self.criterion_cycle
            self.lambda_HPE = opt.lambda_HPE
            self.criterion_HPE = torch.nn.MSELoss()

    def evaluate_HPE(self, fake, real):
        annotated = real.view(*real.shape[:-2], -1).max(dim=-1)[0] > 0
        loss = self.criterion_HPE(fake * annotated[:, :, None, None], real) * self.lambda_HPE
        return loss

    def backward_G(self, backward=True):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_cycle = self.lambda_cycle
        lambda_adv = self.lambda_adversarial

        # Adversarial loss
        self.loss_adv = 0.
        loss_adv_1 = 0.
        loss_adv_2 = 0.

        loss_adv_1 += self.criterion_GAN(self.netD_PB(torch.cat((self.fake_P2, self.input_BP2), 1)), True)
        loss_adv_2 += self.criterion_GAN(self.netD_PB(torch.cat((self.fake_P1, self.input_BP1), 1)), True)

        self.loss_adv = (loss_adv_1 + loss_adv_2) * lambda_adv

        # Identity loss
        self.loss_idt = 0.
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.loss_idt += self.criterion_idt(self.idt_P1, self.input_P1)[0] * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.loss_idt += self.criterion_idt(self.idt_P2, self.input_P2)[0] * lambda_idt

        # HPE Loss
        self.loss_HPE = 0.
        self.fake_BP1 = self.netHPE(self.fake_P1)
        self.fake_BP2 = self.netHPE(self.fake_P2)
        if self.lambda_HPE:
            self.real_BP1 = self.netHPE(self.input_P1)
            self.real_BP2 = self.netHPE(self.input_P2)
            # self.loss_HPE += self.evaluate_HPE(self.fake_BP1, self.real_BP1)
            # self.loss_HPE += self.evaluate_HPE(self.fake_BP2, self.real_BP2)
            self.loss_HPE += self.criterion_HPE(self.fake_BP1, self.real_BP1) * self.lambda_HPE
            self.loss_HPE += self.criterion_HPE(self.fake_BP2, self.real_BP2) * self.lambda_HPE

        self.loss_cycle = 0.
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle += self.criterion_cycle(self.rec_P1, self.input_P1)[0] * lambda_cycle
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle += self.criterion_cycle(self.rec_P2, self.input_P2)[0] * lambda_cycle

        # combined loss and calculate gradients
        self.loss_G = self.loss_adv + self.loss_cycle + self.loss_idt + self.loss_HPE
        if backward:
            self.loss_G.backward()

    def optimize_parameters(self, backward=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(backward=backward)  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

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
        ret_errors = TransferCycleModel.get_current_errors(self)
        if self.lambda_HPE:
            ret_errors['HPE'] = self.loss_HPE.item()

        return ret_errors

    def get_current_visuals(self):
        if not self.gen_final_hpe:
            netHPE = self.netHPE
            if isinstance(netHPE, torch.nn.DataParallel):
                netHPE = netHPE.module
            with torch.no_grad():
                self.fake_BP1 = netHPE.generate_final_bps(self.fake_BP1[:1], self.input_P2[:1])
                self.fake_BP2 = netHPE.generate_final_bps(self.fake_BP2[:1], self.input_P1[:1])
        return TransferCycleModel.get_current_visuals(self)
