import torch

from .PoseCycleNet import TransferCycleModel
from .hpe.openpose import get_pose_net
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss, PerceptualLoss
from losses.color_loss import ColorLossScale
from util.image_pool import ImagePoolPast


class UCCPT(TransferCycleModel):
    def name(self):
        return 'UCCPT'

    def initialize(self, opt):
        opt.with_D_simple = False
        opt.with_D_PP = False
        opt.with_D_PB = True

        opt.use_mask = False

        opt.which_model_netG = "PATN"

        if opt.pool_size:
            opt.pool_size = opt.batchSize

        TransferCycleModel.initialize(self, opt)

        self.model_names.append('netHPE')
        self.netHPE = get_pose_net()

        if self.isTrain:
            self.percep_model = self.netHPE.get_feature_extractor(opt.perceptual_layers, self.gpu_ids)
            self.criterion_cycle = L1_plus_perceptualLoss(1., 0.5, opt.perceptual_layers, self.gpu_ids,
                                                          percep_is_l1=True,
                                                          submodel=self.percep_model)
            self.criterion_idt = self.criterion_cycle
            self.lambda_HPE = opt.lambda_HPE
            self.criterion_HPE = torch.nn.MSELoss()

            self.fake_PB_pool = ImagePoolPast(opt.pool_size)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_P2 = self.netG([self.input_P1, self.input_BP1, self.input_BP2])  # G_A(A)
        self.rec_P1 = self.netG([self.fake_P2, self.input_BP2, self.input_BP1])  # G_B(G_A(A))

        self.fake_P1 = self.netG([self.input_P2, self.input_BP2, self.input_BP1])  # G_B(B)
        self.rec_P2 = self.netG([self.fake_P1, self.input_BP1, self.input_BP2])  # G_A(G_B(B))

        if not self.isTrain or self.lambda_identity:
            self.idt_P1 = self.netG([self.input_P1, self.input_BP1, self.input_BP1])
            self.idt_P2 = self.netG([self.input_P2, self.input_BP2, self.input_BP2])

        self.fake_BP1, self.fake_BP1_feats = self.netHPE(self.fake_P1)
        self.fake_BP2, self.fake_BP2_feats = self.netHPE(self.fake_P2)

    def test(self):
        with torch.no_grad():
            self.fake_P2 = self.netG([self.input_P1, self.input_BP1, self.input_BP2])  # G_A(A)
            if self.use_mask:
                self.fake_P2 *= self.get_mask(self.input_BP2)

            self.fake_P1 = self.netG([self.input_P2, self.input_BP2, self.input_BP1])  # G_B(B)
            if self.use_mask:
                self.fake_P1 *= self.get_mask(self.input_BP1)

            self.fake_BP1 = self.netHPE(self.fake_P1)[0]
            self.fake_BP2 = self.netHPE(self.fake_P2)[0]
            self.real_BP1 = self.netHPE(self.input_P1)[0]
            self.real_BP2 = self.netHPE(self.input_P2)[0]

    def evaluate_HPE(self, feats_out, feats_in):
        loss = 0
        for i in range(len(feats_in)):
            loss += self.criterion_HPE(feats_in[i], feats_out[i])

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
        if self.lambda_HPE:
            self.real_BP1, self.real_BP1_feats = self.netHPE(self.input_P1)
            self.real_BP2, self.real_BP2_feats = self.netHPE(self.input_P2)
            self.loss_HPE += self.evaluate_HPE(self.fake_BP1_feats, self.real_BP1_feats) * self.lambda_HPE
            self.loss_HPE += self.evaluate_HPE(self.fake_BP2_feats, self.real_BP2_feats) * self.lambda_HPE

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

        # D_BP
        for i in range(self.opt.DG_ratio):
            self.optimizer_D_PB.zero_grad()
            self.backward_D_PB(backward=backward)
            self.optimizer_D_PB.step()

    def get_current_errors(self):
        ret_errors = TransferCycleModel.get_current_errors(self)
        if self.lambda_HPE:
            ret_errors['HPE'] = self.loss_HPE.item()

        return ret_errors
