import os
import sys
import glob

import pandas as pd

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader

from utils import get_fid, get_inception_score, ImageDatasetMulti
from metrics.metrics_ssim_market import ssim_score
from metrics.cal_PCKh import get_pckh_from_hpe


def get_multi_annotations(annot_paths):
    return pd.concat([pd.read_csv(ap, sep=':') for ap in annot_paths])


def eval_dataset(img_dirs):
    image_loader = DataLoader(ImageDatasetMulti(img_dirs), shuffle=False)

    print('Loaded images : ', len(image_loader))

    name = ''
    if "synthe" in img_dirs[0]:
        annot_paths = (
            'dataset/synthe_dripe/synthe-annotation-train.csv',
            'dataset/synthe_dripe/synthe-annotation-test.csv'
        )
        target_annotation = get_multi_annotations(annot_paths)
        gt_size = (240, 320)
        name = 'synthe'
    elif "draiver" in img_dirs[0]:
        annot_paths = (
            'dataset/draiver_data/draiver-annotation_train.csv',
            'dataset/draiver_data/draiver-annotation_test.csv'
        )
        target_annotation = get_multi_annotations(annot_paths)
        gt_size = (192, 256)
        name = 'draiver'
    elif "fashion" in img_dirs[0]:
        annot_paths = (
            'dataset/fashion_data/fashion-resize-annotation-test.csv',
            'dataset/fashion_data/fashion-resize-annotation-test.csv'
        )
        target_annotation = get_multi_annotations(annot_paths)
        gt_size = (256, 172)
        name = 'fashion'
    elif "market" in img_dirs[0]:
        annot_paths = (
            'dataset/market_data/market-annotation-test.csv',
            'dataset/market_data/market-annotation-test.csv'
        )
        target_annotation = get_multi_annotations(annot_paths)

        gt_size = (128, 64)
        name = 'market'
    elif "sviro" in img_dirs[0]:
        target_annotation = None
        gt_size = (640, 960)
        name = 'sviro'
    else:
        raise ValueError('Dataset not implemented')

    print('Dataset: ', name)
    print('Paths:\n', img_dirs)
    print('\nDataset images...')
    IS_input = get_inception_score(image_loader)
    print(f"IS input: {IS_input[0]}, std: {IS_input[1]}")

    from models.hpe.simple_bl import get_pose_net
    op = get_pose_net()
    PCKs_input = get_pckh_from_hpe(img_loader=image_loader, hpe_net=op, target_annotation=target_annotation,
                                   gt_size=gt_size)
    print(f'\nPCKh input: {PCKs_input[0] * 100:.2f}% ({PCKs_input[1]}/{PCKs_input[2]} )')


def get_last_dir(dpath):
    last_dir = ''
    last_mtime = 0
    for fold in glob.glob(os.path.join(dpath, '*/')):
        mtime = os.path.getmtime(fold)
        if mtime > last_mtime:
            last_mtime = mtime
            last_dir = fold

    return last_dir


def get_args():
    img_dirs = []
    args = sys.argv[1:].copy()
    if len(args):
        img_dirs = args

    return img_dirs


if __name__ == '__main__':
    eval_dataset(get_args())
