import os
import sys
import glob

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader

from utils import get_fid, get_inception_score, ImageDatasetSplit
from tool.metrics_ssim_market import ssim_score
from tool.cal_PCKh import get_pckh


def get_metrics(results_dir, idx_fake):
    img_dir = os.path.join(results_dir, 'images')

    target_images_np = ImageDatasetSplit(img_dir, img_idx=2, transform=lambda x: x)
    generated_images_np = ImageDatasetSplit(img_dir, img_idx=idx_fake, transform=lambda x: x)

    source_images_loader = DataLoader(ImageDatasetSplit(img_dir, img_idx=0))
    target_images_loader = DataLoader(ImageDatasetSplit(img_dir, img_idx=2))
    generated_images_loader = DataLoader(ImageDatasetSplit(img_dir, img_idx=idx_fake))

    print('\nInput images...')
    IS_input = get_inception_score(target_images_loader)
    print(f"IS input: {IS_input[0]}, std: {IS_input[1]}")

    print('\nInput generated....')
    IS_output = get_inception_score(generated_images_loader)
    print(f"IS output: {IS_output[0]}, std: {IS_output[1]}")

    print('\nFID...')
    FID = get_fid(generated_images_loader, gt_loader=target_images_loader)
    print("FID: ", FID)

    PCKs = get_pckh(results_dir)
    print(f'\nPCKh: {PCKs[0] * 100:.2f}% ({PCKs[1]}/{PCKs[2]} )')

    print("\nCompute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images_np, target_images_np)
    print("SSIM score %s" % structured_score)


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
    idx_fake = 4

    results_dir = './results/market_APS'

    args = sys.argv[1:].copy()
    if len(args):
        results_dir = f'./results/{args[0]}'
    if len(args) > 1:
        idx_fake = int(args[2])

    results_dir = get_last_dir(results_dir)

    return results_dir, idx_fake


if __name__ == '__main__':
    get_metrics(*get_args())
