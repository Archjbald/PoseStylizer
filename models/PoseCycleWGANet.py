import torch
from torch.autograd import Variable, grad
from .base_model import BaseModel
from . import networks

from util.image_pool import ImagePool
from .PoseCycleHPENet import TransferCycleHPEModel


class TransferCycleWGANModel(TransferCycleHPEModel, BaseModel):
    def name(self):
        return 'TransferCycleWGANModel'

    def initialize(self, opt):
        opt.with_D_simple = False
        opt.with_D_PP = False
        opt.with_D_PB = False

        TransferCycleHPEModel.initialize(self, opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            if opt.dataset == 'fashion':
                img_size = (256, 176)
            else:
                raise NotImplementedError('WGAN not implemented for this dataset')

            self.netD = networks.define_D(opt.P_input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          not opt.no_dropout_D,
                                          n_downsampling=opt.D_n_downsampling,
                                          linear_size=img_size)

            self.fake_pool = ImagePool(opt.pool_size)

            self.model_names.append('netD')
            self.netD_list.append(self.netD)

            if opt.continue_train:
                self.load_network(self.netD, 'netD', opt.which_epoch)

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_D)
            self.schedulers.append(networks.get_scheduler(self.optimizer_D, opt))

    # D: take(P, B) as input
    def backward_D(self, backward=True, gp=True):
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
            loss = loss_real + loss_fake + 10 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss_D += loss

        self.loss_D = loss_D / len(pairs) * self.lambda_GAN
        if backward:
            self.loss_D.backward()

    def backward_G(self, backward=True):
        TransferCycleHPEModel.backward_G(self, backward=False)
        # Adversarial loss
        self.loss_adv = 0.
        loss_adv_1 = - self.netD(self.fake_P2).mean()
        self.loss_adv += loss_adv_1
        loss_adv_2 = - self.netD(self.fake_P1).mean()
        self.loss_adv += loss_adv_2
        self.loss_adv /= 2.

        self.loss_G += self.loss_adv * self.lambda_adversarial
        if backward:
            self.loss_G.backward()

    def optimize_parameters(self, backward=True, c=0.01):
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
        for i in range(self.opt.DG_ratio):
            self.optimizer_D.zero_grad()
            self.backward_D(backward=backward)
            self.optimizer_D.step()
            for p in self.netD.parameters():
                p.data.clamp_(-c, c)

    def get_current_errors(self):
        ret_errors = TransferCycleHPEModel.get_current_errors(self)

        return ret_errors
