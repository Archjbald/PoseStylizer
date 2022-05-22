import time
import os
import pickle
from collections import OrderedDict
from argparse import Namespace
from pytorch_gan_metrics import get_inception_score

import torch

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.util import avg_dic, debug_gpu_memory, get_gpu_memory
from util.random_seed import seed_all
from test import set_test_opt


# os.environ['GPU_DEBUG'] = '0'
# from util.gpu_profile import gpu_profile


def train(opt, model, train_dataset, val_dataset):
    dataset_size = len(train_dataset.dataloader)
    print('#training images = %d' % dataset_size)

    visualizer = Visualizer(opt)

    # log opts
    print('-' * 17, ' Options ', '-' * 17)
    opt_keys = list(opt.__dict__.keys())
    opt_keys.sort()
    opt_msg = ''
    for k in opt_keys:
        opt_msg += f'{k}: {opt.__dict__[k]}\n'
    print(opt_msg)
    with open(visualizer.log_name.replace('loss_log.txt', 'options.txt'), 'w') as f:
        f.write(opt_msg)
    print('-' * 43)

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
    best_IS = 0
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
                if not opt.continue_train:
                    print('saving the initialized model')
                    model.save('init', epoch, total_steps)

            model.set_input(data)
            model.optimize_parameters()

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

            # debug_gpu_memory(model)
            # print(get_gpu_memory())

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

        # save latest model
        print('saving the latest model (epoch %d, total_steps %d)' %
              (epoch, total_steps))
        model.save('latest', epoch, total_steps)

        # save epoch model
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save(epoch, epoch + 1, total_steps)

        # Validation
        if opt.backward in ['basic'] and epoch % opt.val_epoch_freq == 0:
            val_errors = {}
            fakes = []
            for v, val_data in enumerate(val_dataset):
                with torch.no_grad():
                    model.set_input(val_data)
                    model.optimize_parameters(backward=False)
                iter_errors = model.get_current_errors()
                val_errors = avg_dic(val_errors, iter_errors, v)
                fakes.append(model.fake_P1.clone().cpu())
                if v < 5:
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result=True,
                                                       lbls=["val", str(v)])
            current_IS = get_inception_score(torch.cat(fakes))
            val_errors["IS_val"] = current_IS[0]
            val_errors["IS_std"] = current_IS[1]
            visualizer.print_current_errors(epoch, epoch_iter, val_errors, t, val=True)

            if current_IS[0] > best_IS:
                print('saving the best model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('best', epoch, total_steps)
                best_IS = current_IS[0]

            del fakes

        # print time used
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()


def main():
    opt = TrainOptions().parse()

    seed_all(opt.seed)
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
