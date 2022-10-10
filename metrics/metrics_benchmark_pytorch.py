import os
import sys
import glob

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader

from utils import get_fid, get_inception_score, ImageDatasetSplit
from metrics.metrics_ssim_market import ssim_score
from metrics.cal_PCKh import get_pckh_from_hpe


def get_metrics(results_dir, idx_fake):
    from models.hpe.simple_bl import get_pose_net
    op = get_pose_net()

    img_dir = os.path.join(results_dir, 'images')

    source_images_np = ImageDatasetSplit(img_dir, img_idx=0, transform=lambda x: x)
    target_images_np = ImageDatasetSplit(img_dir, img_idx=2, transform=lambda x: x)
    generated_images_np = ImageDatasetSplit(img_dir, img_idx=idx_fake, transform=lambda x: x)

    source_images_loader = DataLoader(ImageDatasetSplit(img_dir, img_idx=0), shuffle=False)
    target_images_loader = DataLoader(ImageDatasetSplit(img_dir, img_idx=2), shuffle=False)
    generated_images_loader = DataLoader(ImageDatasetSplit(img_dir, img_idx=idx_fake), shuffle=False)
    cycle_images_loader = None
    if generated_images_np.len_img > idx_fake + 1:
        idx_cycle = idx_fake + (2 if idx_fake + 3 < generated_images_np.len_img else 1)
        cycle_images_loader = DataLoader(ImageDatasetSplit(img_dir, img_idx=idx_cycle), shuffle=False)
        cycle_images_np = ImageDatasetSplit(img_dir, img_idx=idx_cycle, transform=lambda x: x)

    print('Loaded images : ', len(generated_images_loader))

    if "synthe" in results_dir:
        target_annotation = 'dataset/synthe_dripe/synthe-annotation-test.csv'
    elif "draiver" in results_dir:
        target_annotation = 'dataset/draiver/draiver-annotation_test.csv'
    elif "fashion" in results_dir:
        target_annotation = 'dataset/fashion_data/fashion-resize-annotation-test.csv'
    elif "market" in results_dir:
        target_annotation = 'dataset/market_data/market-annotation-test.csv'
    else:
        raise ValueError('Dataset not implemented')

    print(f'Annotation file : {target_annotation}, {os.path.isfile(target_annotation)}')

    print('\nInput images...')
    IS_input = get_inception_score(target_images_loader)
    print(f"IS input: {IS_input[0]}, std: {IS_input[1]}")

    print('\nInput generated....')
    IS_output = get_inception_score(generated_images_loader)
    print(f"IS output: {IS_output[0]}, std: {IS_output[1]}")

    if cycle_images_loader:
        IS_output_2 = get_inception_score(cycle_images_loader)
        print(f"IS output 2: {IS_output_2[0]}, std: {IS_output_2[1]}")

    print('\nFID...')
    FID = get_fid(generated_images_loader, gt_loader=target_images_loader)
    print("FID: ", FID)

    if cycle_images_loader:
        FID_2 = get_fid(cycle_images_loader, gt_loader=source_images_loader)
        print("FID_2: ", FID_2)

    # PCKs = get_pckh_from_dir(results_dir)
    # print(f'\nPCKh: {PCKs[0] * 100:.2f}% ({PCKs[1]}/{PCKs[2]} )')

    PCKs_input = get_pckh_from_hpe(img_loader=target_images_loader, hpe_net=op, target_annotation=target_annotation)
    print(f'\nPCKh input: {PCKs_input[0] * 100:.2f}% ({PCKs_input[1]}/{PCKs_input[2]} )')

    PCKs_output = get_pckh_from_hpe(img_loader=generated_images_loader, hpe_net=op, target_annotation=target_annotation)
    print(f'\nPCKh output: {PCKs_output[0] * 100:.2f}% ({PCKs_output[1]}/{PCKs_output[2]} )')

    print("\nCompute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images_np, target_images_np)
    print("SSIM score %s" % structured_score)

    if cycle_images_loader:
        PCKs_output_2 = get_pckh_from_hpe(img_loader=cycle_images_loader, hpe_net=op,
                                          target_annotation=target_annotation)
        print(f'\nPCKh_2 output: {PCKs_output_2[0] * 100:.2f}% ({PCKs_output_2[1]}/{PCKs_output_2[2]} )')

        print("\nCompute structured similarity score (SSIM)...")
        structured_score_2 = ssim_score(cycle_images_np, source_images_np)
        print("SSIM score_2 %s" % structured_score_2)


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
