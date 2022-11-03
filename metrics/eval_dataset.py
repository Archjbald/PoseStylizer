import os
import sys
import glob

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader

from utils import get_fid, get_inception_score, ImageDatasetMulti
from metrics.metrics_ssim_market import ssim_score
from metrics.cal_PCKh import get_pckh_from_hpe


def eval_dataset(img_dirs):
    from models.hpe.simple_bl import get_pose_net
    # op = get_pose_net()

    image_loader = ImageDatasetMulti(img_dirs)

    print('Loaded images : ', len(image_loader))

    if "synthe" in img_dirs[0]:
        target_annotation = 'dataset/synthe_dripe/synthe-annotation-test.csv'
        gt_size = (240, 320)
    elif "draiver" in img_dirs[0]:
        target_annotation = 'dataset/draiver/draiver-annotation_test.csv'
        gt_size = (192, 256)
    elif "fashion" in img_dirs[0]:
        target_annotation = 'dataset/fashion_data/fashion-resize-annotation-test.csv'
        gt_size = (256, 172)
    elif "market" in img_dirs[0]:
        target_annotation = 'dataset/market_data/market-annotation-test.csv'
        gt_size = (128, 64)
    else:
        raise ValueError('Dataset not implemented')

    print('\nDataset images...')
    IS_input = get_inception_score(image_loader)
    print(f"IS input: {IS_input[0]}, std: {IS_input[1]}")


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
