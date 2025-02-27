from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=100,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=100,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--val_epoch_freq', type=int, default=10,
                                 help='frequency of calculating val loss at the end of epochs')

        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for perceptual L1 loss')
        self.parser.add_argument('--lambda_GAN', type=float, default=5.0, help='weight of GAN loss')

        self.parser.add_argument('--epoch_size', type=int, default=4000,
                                 help='Number of image pooled in the dataset for each epcoh')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=100,
                                 help='multiply by a gamma every lr_decay_iters iterations')

        self.parser.add_argument('--L1_type', type=str, default='origin', choices=['origin', 'l1_plus_perL1'],
                                 help='use which kind of L1 loss. (origin|l1_plus_perL1)')
        self.parser.add_argument('--perceptual_layers', type=int, default=3,
                                 help='index of vgg layer for extracting perceptual features.')
        self.parser.add_argument('--percep_is_l1', type=int, default=1, help='type of perceptual loss: l1 or l2')
        self.parser.add_argument('--no_dropout_D', action='store_true', help='no dropout for the discriminator')
        self.parser.add_argument('--DG_ratio', type=int, default=1,
                                 help='how many times for D training after training G once')

        # CUT
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--patch_sizes', type=str, default='3',
                                 help='compute NCE loss on which layers')
        self.parser.add_argument('--in_mask', action='store_true', help='pick random patch in the body mask')
        self.parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')

        # Cycle
        self.parser.add_argument('--lambda_cycle', type=float, default=10.,
                                 help='the cycle loss')
        self.parser.add_argument('--lambda_identity', type=float, default=0.8,
                                 help='the "identity preservation loss"')
        self.parser.add_argument('--lambda_adversarial', type=float, default=1.,
                                 help='the "adversarial loss"')
        self.parser.add_argument('--lambda_HPE', type=float, default=0.,
                                 help='the generated keypoint loss'),
        self.parser.add_argument('--lambda_patch', type=float, default=0.,
                                 help='the generated keypoint loss')
        self.parser.add_argument('--nb_patch', type=int, default=1,
                                 help='number of patches for patch loss')

        self.parser.add_argument('--use_mask', action='store_true',
                                 help='mask background')


        self.isTrain = True

    def parse(self):
        BaseOptions.parse(self)

        if self.opt.debug:
            self.opt.epoch_size = min(self.opt.epoch_size, 20)

        if self.opt.epoch_size % self.opt.batchSize:
            self.opt.epoch_size = (self.opt.epoch_size // self.opt.batchSize + 1) * self.opt.batchSize

        self.opt.patch_sizes = [int(k) for k in self.opt.patch_sizes.split(',')]

        return self.opt
