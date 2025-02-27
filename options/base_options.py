import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True,
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--dataset', type=str, default='market', help='dataset name')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='resnet', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='PATN', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='blocks used in D')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='', help='chooses how datasets are loaded')
        self.parser.add_argument('--random', action='store_true', help='if true, randomly shuffle input images')
        self.parser.add_argument('--shuffle', action='store_true', help='if true, shuffle actors for training')
        self.parser.add_argument('--extend', action='store_true', help='use extended version of datasets (27 joints)')
        self.parser.add_argument('--debug', action='store_true', help='debug mode')

        self.parser.add_argument('--model', type=str, default='', help='chooses which model to use')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint', help='models are saved here')
        self.parser.add_argument('--log_dir', type=str, default='./logs', help='logs are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--drop_layers', type=int, default=0, help='number of layers that adopts dropout')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--no_rotate', action='store_true',
                                 help='if specified, do not rotate the images for data augmentation')

        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.parser.add_argument('--P_input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--BP_input_nc', type=int, default=18, help='# of input image channels')
        self.parser.add_argument('--padding_type', type=str, default='reflect', help='# of input image channels')
        self.parser.add_argument('--pairLst', type=str, default='', help='market pairs')

        self.parser.add_argument('--with_D_simple', type=int, default=1, help='use simple D')
        self.parser.add_argument('--with_D_PP', type=int, default=1, help='use D to judge P and P is pair or not')
        self.parser.add_argument('--with_D_PB', type=int, default=1, help='use D to judge P and B is pair or not')

        # down-sampling times
        self.parser.add_argument('--G_n_downsampling', type=int, default=5, help='down-sampling blocks for generator')
        self.parser.add_argument('--D_n_downsampling', type=int, default=2,
                                 help='down-sampling blocks for discriminator')

        self.initialized = True
        # CUT options
        self.parser.add_argument('--backward', type=str, default='basic',
                                 choices=['basic', 'cut', 'cycle', 'better_cycle', 'cycle_hpe', 'cycle_wgan'],
                                 help='choose between classic APS or CUT backward method for generator')
        self.parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        self.parser.add_argument('--init_gain', type=float, default=0.02,
                                 help='scaling factor for normal, xavier and orthogonal.')

        self.parser.add_argument('--fake_bp_cycle', action='store_true',
                                 help='Use predicted BP for cycle instead of ground truth')

        self.parser.add_argument('--no_nce_idt', action='store_true',
                                 help='not use NCE loss for identity mapping: NCE(G(Y), Y))')

        self.parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16',
                                 help='compute NCE loss on which layers')
        self.parser.add_argument('--nce_includes_all_negatives_from_minibatch', action='store_true',
                                 help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        self.parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                                 help='how to downsample the feature map')
        self.parser.add_argument('--netF_nc', type=int, default=256)
        self.parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        self.parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        self.parser.add_argument('--flip_equivariance', action='store_true',
                                 help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        self.parser.add_argument('--use_transfer_layer', action='store_true',
                                 help='Use transfer layer in the generator')

        self.parser.set_defaults(pool_size=0)

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)
        if self.opt.extend:
            self.opt.BP_input_nc = 27

        # Set default parameters for CUT and FastCUT
        if self.opt.CUT_mode.lower() == "cut":
            self.parser.set_defaults(no_nce_idt=False, lambda_NCE=1.0)
        elif self.opt.CUT_mode.lower() == "fastcut":
            self.parser.set_defaults(
                no_nce_idt=True, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(self.opt.CUT_mode)


        # Path
        if not self.opt.pairLst:
            self.opt.pairLst = os.path.join(self.opt.dataroot,
                                            f'{self.opt.dataset}-pairs-{"train" if self.opt.isTrain else "test"}.csv')

        if self.opt.shuffle and "shuffle" not in self.opt.pairLst:
            self.opt.pairLst = self.opt.pairLst.replace('.csv', '-shuffle.csv')

        # if self.opt.debug and "small" not in self.opt.pairLst:
        #     self.opt.pairLst = self.opt.pairLst.replace('.csv', '-small.csv')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
