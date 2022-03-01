import time
import numpy as np
import os
import sys
import pickle
from collections import OrderedDict
from argparse import Namespace
import gc

import torch

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.util import avg_dic, get_gpu_memory
from test import set_test_opt

# os.environ['GPU_DEBUG'] = '0'
# from util.gpu_profile import gpu_profile


def train(opt, model, train_dataset, val_dataset):
    dataset_size = len(train_dataset.dataloader)
    print('#training images = %d' % dataset_size)

    visualizer = Visualizer(opt)

    info_dir = os.path.join(opt.checkpoints_dir, opt.name)
    infoname = '%s.pkl' % (opt.which_epoch)
    infoname = os.path.join(info_dir, infoname)
    if opt.continue_train and os.path.exists(infoname):
        print('Loaded epoch and total_steps')
        file = open(infoname, 'rb')
        info = pickle.load(file)
        file.close()
        epoch_count = info['epoch']
        total_steps = info['total_steps']
    else:
        epoch_count = opt.epoch_count
        total_steps = 0

    print("Start epoch: ", epoch_count)
    print("Total steps: ", total_steps)

    for steps in range(epoch_count - 1):
        for scheduler in model.schedulers:
            scheduler.step()

    with open('garbage.log', 'w') as f:
        f.write(''.join(['=' * 10, 'Epoch 0', "=" * 10, '\n']))

    stat_errors = OrderedDict([('count', 0)])
    for epoch in range(epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(train_dataset):
            with open('garbage.log', 'a') as f:
                f.write(''.join(['*' * 10, f'Iter {i}', "*" * 10, '\n']))

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += 1
            epoch_iter += 1

            if epoch == epoch_count and i == 0:
                if opt.backward == 'cut':
                    model.data_dependent_initialize(data)
                model.parallelize()

            model.set_input(data)


            model.optimize_parameters()
            """
            # stat errors
            current_errors = model.get_current_errors()
            stat_errors['count'] += 1
            for key in current_errors.keys():
                if key in stat_errors:
                    stat_errors[key] += current_errors[key]
                else:
                    stat_errors[key] = current_errors[key]

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = time.time() - iter_start_time
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

            # save latest model
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest', epoch, total_steps)

            """

            print("Free memory: ", get_gpu_memory())
            attribs = {k: v.nelement() * v.element_size() for k, v in model.__dict__.items() if
                       isinstance(v, torch.Tensor)}

            with open('garbage.log', 'a') as f:
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            f.write(f'{type(obj)}: {obj.size()}\n')
                    except:
                        pass

                f.write('%'*20)
                for n, t in model.named_parameters():
                    f.write(f'{n}: {t.shape}\n')

                f.write('%' * 20)
                for m in [model, model.netG]:
                    for n, t in m.__dict__.items():
                        if torch.is_tensor(t):
                            f.write(f'{n}: {t.shape}\n')

            """
            if i < len(train_dataset) - 1:
                model.cleanse()
            """

            if i > 5:
                sys.exit(0)

        t = time.time() - iter_start_time
        for key in stat_errors.keys():
            if not key == 'count':
                stat_errors[key] /= stat_errors['count']
        visualizer.print_current_errors(epoch, epoch_iter, stat_errors, t)
        if opt.display_id > 0:
            visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, stat_errors)
        stat_errors = OrderedDict([('count', 0)])

        # save images
        save_result = False
        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        # save epoch model
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest', epoch + 1, total_steps)
            model.save(epoch, epoch + 1, total_steps)

        # print time used
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # Validation
        if opt.backward in ['basic'] and epoch % opt.val_epoch_freq == 0:
            val_errors = {}
            for v, val_data in enumerate(val_dataset):
                with torch.no_grad():
                    print(len(val_data))
                    model.set_input(val_data)
                    model.optimize_parameters(backward=False)
                iter_errors = model.get_current_errors()
                val_errors = avg_dic(val_errors, iter_errors, v)
                model.cleanse()
            visualizer.print_current_errors(epoch, epoch_iter, val_errors, t, val=True)

        model.cleanse()

    model.update_learning_rate()


def main():
    opt = TrainOptions().parse()

    train_data_loader = CreateDataLoader(opt)
    train_dataset = train_data_loader.load_data()

    val_size = round(100 / opt.batchSize) * opt.batchSize
    val_opt = set_test_opt(Namespace(**vars(opt)), max_dataset_size=val_size)
    val_data_loader = CreateDataLoader(val_opt)
    val_dataset = val_data_loader.load_data()

    model = create_model(opt)

    train(opt, model, train_dataset, val_dataset)


if __name__ == '__main__':
    # sys.settrace(gpu_profile)
    main()
