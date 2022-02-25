import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import OrderedDict


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids, epoch, total_steps):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_infoname = '%s.pkl' % (epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        save_infoname = os.path.join(self.save_dir, save_infoname)
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

        info = {'epoch': epoch, 'total_steps': total_steps}
        filehandler = open(save_infoname, "wb")
        pickle.dump(info, filehandler)
        filehandler.close()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        if isinstance(network, torch.nn.DataParallel):
            network = network.module

        if os.path.exists(save_path):
            named_params = [n for n, _ in network.named_parameters()]
            checkpoint = torch.load(save_path)
            checkpoint = OrderedDict(
                [(k.replace('module.', '') if k.replace('module.', '') in named_params else k, v) for k, v in
                 checkpoint.items()])
            try:
                network.load_state_dict(checkpoint)
            except RuntimeError as err:
                msg = str(err).split('\n')
                missing = msg[1].split(': ')[1].split(', ')
                unexpect = msg[2].split(': ')[1].split(', ')

                raise err

            print("Found checkpoints. Network loaded.")
        else:
            print("Not found checkpoints. Network from scratch.")

    def parallelize(self):
        if len(self.opt.gpu_ids) <= 1:
            pass
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                setattr(self, name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
