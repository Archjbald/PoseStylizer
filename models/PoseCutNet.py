import numpy as np
import torch
from collections import OrderedDict

from .base_model import BaseModel
from util.image_pool import ImagePool
from . import networks
import util.util as util

from losses.patchnce import PatchNCELoss


class TransferCUTModel(BaseModel):
    """
    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def name(self):
        return 'TransferCUTModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt

        self.nce_nb_layers = opt.G_n_downsampling + 2
        self.model_names = ['netG', 'netF']

        # define networks (both generator and discriminator)
        input_nc = [opt.P_input_nc, opt.BP_input_nc, opt.BP_input_nc]

        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                      opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                      self.gpu_ids, n_downsampling=opt.G_n_downsampling, opt=opt)

        self.netF = networks.define_F(opt.netF, opt.init_type, opt.init_gain, self.gpu_ids, opt)

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

            self.criterionNCE = []
            for _ in range(self.nce_nb_layers):
                self.criterionNCE.append(PatchNCELoss(opt))

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
        self.input_P1, self.input_BP1 = input['P1'], input['BP1']
        self.input_P2, self.input_BP2 = input['P2'], input['BP2']
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

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["P1"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.input_P1 = self.input_P1[:bs_per_gpu]
        self.input_BP1 = self.input_BP1[:bs_per_gpu]
        self.input_P2 = self.input_P2[:bs_per_gpu]
        self.input_BP2 = self.input_BP2[:bs_per_gpu]
        if self.opt.dataset_mode in ['keypoint_segmentation']:
            self.input_MP1 = self.input_MP1[:bs_per_gpu]
            self.input_MP2 = self.input_MP2[:bs_per_gpu]

        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            if self.opt.with_D_PP:  # calculate gradients for D
                self.backward_D_PP()
            if self.opt.with_D_PP:
                self.backward_D_PB()
            self.backward_G()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        nbj = self.opt.BP_input_nc

        G_input = [self.input_P1[:, :nbj], self.input_BP1[:, :nbj], self.input_BP2[:, :nbj]]
        if not self.opt.no_nce_idt and self.opt.isTrain:
            G_input = [
                torch.cat((self.input_P1, self.input_P2), dim=0),
                torch.cat((self.input_BP1[:, :nbj], self.input_BP2[:, :nbj]), dim=0),
                torch.cat((self.input_BP1[:, :nbj], self.input_BP2[:, :nbj]), dim=0),
            ]

        if self.opt.dataset_mode == 'keypoint_segmentation':
            G_input.append(self.input_MP1)
        output = self.netG(G_input)

        self.fake_P2 = output[:self.input_P1.size(0)]
        if not self.opt.no_nce_idt:
            self.idt_P2 = output[self.input_P1.size(0):]

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
        nbj = self.opt.BP_input_nc
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_P2, self.input_BP2[:, :nbj]), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_P2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_A
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_A
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_A

        # First, G(A) should fake the discriminator
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss([self.input_P1, self.input_BP1],
                                                    [self.fake_P2, self.input_BP2])
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if not self.opt.no_nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss([self.fake_P2, self.input_BP2],
                                                      [self.idt_P2, self.input_BP2])
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = loss_NCE_both + pair_GANloss
        else:
            pair_loss = loss_NCE_both

        if backward:
            pair_loss.backward()

        self.loss_NCE_both = loss_NCE_both.item()
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.item()
        self.loss_G = pair_loss.item()

    def optimize_parameters(self, backward=True):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.backward_G(backward=backward)
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

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

    def calculate_NCE_loss(self, src, tgt):
        try:
            feat_tgt = self.netG(tgt + [self.input_BP2, ], encode_only=True)
        except TypeError as err:
            print(tgt, self.input_BP2)
            raise err

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_tgt = [torch.flip(fq, [3]) for fq in feat_tgt]

        feat_src = self.netG(src + [self.input_BP2, ], encode_only=True)

        scales = torch.tensor([f.shape[-2:] for f in feat_tgt], device=self.input_P1.device)

        if isinstance(self.netF, torch.nn.DataParallel):
            sample_ids, num_patches = self.netF.module.get_ids_kps(self.input_BP1, self.input_BP2, scales,
                                                                   patch_sizes=self.opt.patch_sizes,
                                                                   num_patches=self.opt.num_patches,
                                                                   in_mask=self.opt.in_mask)
        else:
            sample_ids, num_patches = self.netF.get_ids_kps(self.input_BP1, self.input_BP2, scales,
                                                            patch_sizes=self.opt.patch_sizes,
                                                            num_patches=self.opt.num_patches,
                                                            in_mask=self.opt.in_mask)

        self.opt.num_patches = num_patches
        feat_src_pool, _ = self.netF(feat_src, num_patches=self.opt.num_patches, patch_ids=sample_ids[0])
        feat_tgt_pool, _ = self.netF(feat_tgt, num_patches=self.opt.num_patches, patch_ids=sample_ids[1])

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_tgt_pool, feat_src_pool, self.criterionNCE,
                                             range(self.nce_nb_layers)):
            try:
                loss = crit(f_q, f_k) * self.opt.lambda_NCE
            except RuntimeError as err:
                print('\t'.join([str(q.shape) for q in tgt]), self.input_BP2.shape)
                print('\t'.join([str(q.shape) for q in feat_tgt]))
                print('\t'.join([str(q.shape) for q in feat_tgt_pool]))
                raise err
            total_nce_loss += loss.mean()

        return total_nce_loss / self.nce_nb_layers

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['pair_GANloss'] = self.pair_GANloss

        ret_errors['loss_NCE'] = self.loss_NCE
        if not self.opt.no_nce_idt and self.opt.lambda_NCE > 0.0:
            ret_errors['loss_NCE_Y'] = self.loss_NCE_Y
        ret_errors['loss_NCE_both'] = self.loss_NCE_both
        ret_errors['G'] = self.loss_G

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
        self.save_network(self.netF, 'netF', label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB, 'netD_PB', label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids, epoch, total_steps)
