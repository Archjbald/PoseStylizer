import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time


def test(opt, model, dataset):
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    print("testing samples: %d/%d" % (opt.how_many, len(dataset)))

    for i, data in enumerate(dataset):
        #     print(' process %d/%d img ..'%(i, opt.how_many))
        if i >= opt.how_many:
            break
        model.set_input(data)
        startTime = time.time()
        model.test()
        endTime = time.time()
        visuals = model.get_current_visuals()
        #     visuals = model.get_current_visuals_widerpose()
        img_path = model.get_image_paths()
        img_path = [img_path]
        visualizer.save_images(webpage, visuals, img_path)
        if not i % 100:
            print(i)

    webpage.save()


def set_test_opt(opt, max_dataset_size=None):
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.phase = 'test'
    opt.pairLst = opt.pairLst.replace('train', 'test')
    if max_dataset_size:
        opt.max_dataset_size = max_dataset_size

    return opt


def main():
    opt = TestOptions().parse()
    opt = set_test_opt(opt)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    model = create_model(opt)
    model = model.eval()

    # test
    test(opt, model, dataset)


if __name__ == '__main__':
    main()
