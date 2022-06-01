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
        self.annots_real = {'x': [], 'y': []}
        self.annots_fake = {'x': [], 'y': []}
        self.img_paths = []

    def add_annots(self, real, fake, img_paths=None):
        real_kps, real_v = get_kps(real)
        fake_kps, fake_v = get_kps(fake)

        for k, v, annots in [(real_kps, real_v, self.annots_real), (fake_kps, fake_v, self.annots_fake)]:
            vis = v[:, :, None]
            kps = k * vis + (~vis) * -1
            annots['x'] += kps[:, :, 0].tolist()
            annots['y'] += kps[:, :, 1].tolist()

        if img_paths:
            self.img_paths += img_paths.split('___')

    def save(self, path):
        for name, annots in [('real', self.annots_real), ('fake', self.annots_fake)]:
            annots_csv = "name:keypoints_y:keypoints_x\n"
            annots_csv += '\n'.join(
                [f'{img}:{annots["y"][i]}:{annots["x"][i]}' for i, img in enumerate(self.img_paths)])

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
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

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
        visuals = model.get_current_visuals_test()
        #     visuals = model.get_current_visuals_widerpose()
        img_path = model.get_image_paths()

        hpe_annots.add_annots(model.real_BP1, model.fake_BP1, img_paths=img_path)
        hpe_annots.add_annots(model.real_BP2, model.fake_BP2)

        img_path = [img_path]
        visualizer.save_images(webpage, visuals, img_path)
        if not i % 100:
            print(i)

    webpage.save()
    hpe_annots.save(webpage.web_dir)


def set_test_opt(opt, max_dataset_size=None):
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.phase = 'test'
    opt.isTrain = False
    opt.pairLst = opt.pairLst.replace('train', 'test')
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
