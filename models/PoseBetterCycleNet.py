import torch

from .PoseCycleNet import TransferCycleModel


class TransferBetterCycleModel(TransferCycleModel):

    def name(self):
        return 'TransferBetterCycleModel'

    def initialize(self, opt):
        TransferCycleModel.initialize(self, opt)
        self.lambda_percep = 0.2

    def bacwkard_cycle(self):
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        lambda_idt = self.lambda_identity
        lambda_percep = self.lambda_percep if (self.opt.with_D_PP or self.opt.with_D_PB) else 1.

        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_1 = self.netG([self.input_P1, self.input_BP1, self.input_BP1])
            self.loss_idt_1 = self.criterion_idt(self.idt_1, self.input_P1) * lambda_B * lambda_idt * lambda_percep

            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_2 = self.netG([self.input_P2, self.input_BP2, self.input_BP2])
            self.loss_idt_2 = self.criterion_idt(self.idt_2, self.input_P2) * lambda_A * lambda_idt * lambda_percep

            lambda_percep = 1 - lambda_percep
            lambda_percep /= max(self.opt.with_D_PP + self.opt.with_D_PB, 1)
            if self.opt.with_D_PP:
                with torch.no_grad():
                    feat_real_P1 = self.netD_PP(torch.cat((self.input_P1, self.input_P2), 1), use_sigmoid=False)
                    feat_fake_P1 = self.netD_PP(torch.cat((self.fake_P1, self.input_P2), 1), use_sigmoid=False)
                self.loss_idt_1 += self.criterion_idt(feat_real_P1,
                                                      feat_fake_P1) * lambda_A * lambda_idt * lambda_percep

                with torch.no_grad():
                    feat_real_P2 = self.netD_PP(torch.cat((self.input_P2, self.input_P1), 1), use_sigmoid=False)
                    feat_fake_P2 = self.netD_PP(torch.cat((self.fake_P2, self.input_P1), 1), use_sigmoid=False)
                self.loss_idt_2 += self.criterion_idt(feat_real_P2,
                                                      feat_fake_P2) * lambda_B * lambda_idt * lambda_percep
            if self.opt.with_D_PB:
                with torch.no_grad():
                    feat_real_P1 = self.netD_PB(torch.cat((self.input_P1, self.input_BP1), 1), use_sigmoid=False)
                    feat_fake_P1 = self.netD_PB(torch.cat((self.fake_P1, self.input_BP1), 1), use_sigmoid=False)
                self.loss_idt_1 += self.criterion_idt(feat_real_P1,
                                                      feat_fake_P1) * lambda_A * lambda_idt * lambda_percep

                with torch.no_grad():
                    feat_real_P2 = self.netD_PB(torch.cat((self.input_P2, self.input_BP1), 1), use_sigmoid=False)
                    feat_fake_P2 = self.netD_PB(torch.cat((self.fake_P2, self.input_BP1), 1), use_sigmoid=False)
                self.loss_idt_2 += self.criterion_idt(feat_real_P2,
                                                      feat_fake_P2) * lambda_B * lambda_idt * lambda_percep
        else:
            self.loss_idt_1 = 0
            self.loss_idt_2 = 0

    def backward_G(self, backward=True):
        """Calculate the loss for generators G_A and G_B"""

        lambda_A = self.lambda_A
        lambda_B = self.lambda_B

        # Identity loss
        self.bacwkard_cycle()

        self.loss_G_1 = 0
        self.loss_G_2 = 0
        if self.opt.with_D_PB:
            self.loss_G_GAN_PB_1 = self.criterion_GAN(self.netD_PB(torch.cat((self.fake_P2, self.input_BP2), 1)), True)
            self.loss_G_1 += self.loss_G_GAN_PB_1
            self.loss_G_GAN_PB_2 = self.criterion_GAN(self.netD_PB(torch.cat((self.fake_P1, self.input_BP1), 1)), True)
            self.loss_G_2 += self.loss_G_GAN_PB_2

        if self.opt.with_D_PP:
            self.loss_G_GAN_PP_1 = self.criterion_GAN(self.netD_PP(torch.cat((self.fake_P2, self.input_P1), 1)), True)
            self.loss_G_GAN_PP_2 = self.criterion_GAN(self.netD_PP(torch.cat((self.fake_P1, self.input_P2), 1)), True)
            self.loss_G_1 += self.loss_G_GAN_PP_1
            self.loss_G_2 += self.loss_G_GAN_PP_2

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_1 = self.criterion_cycle(self.rec_P1, self.input_P1) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_2 = self.criterion_cycle(self.rec_P2, self.input_P2) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_1 + self.loss_G_1 + self.loss_cycle_1 + self.loss_cycle_2 + self.loss_idt_1 + self.loss_idt_2
        if backward:
            self.loss_G.backward()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        epoch = scheduler.last_epoch
        progress = epoch / (self.opt.niter + self.opt.niter_decay)
        self.lambda_percep = 0.2 + progress * 0.6
        self.lambda_idt = 0.8 - progress * 0.6
