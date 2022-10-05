import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util.util import get_kps
import time


class HPEAnnots:
    def __init__(self):
        self.annots = {'x': [], 'y': []}
        # Picture format coordinates
        self.img_paths = []

    def add_annots(self, bps, img_paths=None):
        kps, v = get_kps(bps)

        vis = v[:, :, None]
        kps = kps * vis + (~vis) * -1
        self.annots['x'] += kps[:, :, 1].tolist()
        self.annots['y'] += kps[:, :, 0].tolist()

        if img_paths:
            self.img_paths += img_paths

    def save(self, path, name=None):
        annots_csv = "name:keypoints_y:keypoints_x\n"
        annots_csv += '\n'.join(
            [f'{img}:{self.annots["y"][i]}:{self.annots["x"][i]}' for i, img in enumerate(self.img_paths)])

        with open(os.path.join(path, f"annots_{name}.csv"), 'w') as f:
            f.write(annots_csv)


def test(opt, model, dataset):
    visualizer = Visualizer(opt)

    # log opts
    print('-' * 17, ' Options ', '-' * 17)
    opt_keys = list(opt.__dict__.keys())
    opt_keys.sort()
    opt_msg = ''
    for k in opt_keys:
        opt_msg += f'{k}: {opt.__dict__[k]}\n'
    print(opt_msg)
    print('-' * 43)

    hpe_annots = HPEAnnots()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           f"{opt.phase}_{opt.which_epoch}" + ("_shuffle" if "shuffle" in opt.pairLst else ""))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    print("testing samples: %d/%d" % (opt.how_many, len(dataset)))

    for i, data in enumerate(dataset):
        #     print(' process %d/%d img ..'%(i, opt.how_many))
        if i >= opt.how_many:
            break

        if i == 0:
            if opt.backward == 'cut':
                model.data_dependent_initialize(data)
            model.parallelize()

        model.set_input(data)
        startTime = time.time()
        model.test()
        endTime = time.time()
        model.test_D()
        visuals = model.get_current_visuals_test()
        #     visuals = model.get_current_visuals_widerpose()
        img_paths = model.get_image_paths()

        hpe_annots.add_annots(model.input_BP1, img_paths=[ip.split('___')[0] for ip in img_paths])
        hpe_annots.add_annots(model.input_BP2, img_paths=[ip.split('___')[1] for ip in img_paths])

        visualizer.save_images(webpage, visuals, img_paths)
        if not i % 100:
            print(i)

    webpage.save()
    hpe_annots.save(webpage.web_dir, name="real")


def set_test_opt(opt, max_dataset_size=None):
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.phase = 'test'
    opt.isTrain = False
    opt.pairLst = os.path.join(os.path.split(opt.pairLst)[0], os.path.split(opt.pairLst)[1].replace('train', 'test'))
    opt.random = False
    if max_dataset_size:
        opt.max_dataset_size = max_dataset_size

    return opt


def main(opt=None):
    if opt is None:
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
