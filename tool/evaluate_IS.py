from pytorch_gan_metrics import get_inception_score_from_directory, get_fid, get_inception_score
import numpy as np
import os
import skimage
import sys
import torch


def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []

    names = []
    img_list = os.listdir(images_folder)
    for img_name in img_list:
        img = skimage.io.imread(os.path.join(images_folder, img_name))
        w = int(img.shape[1] / LEN_IMG)  # h, w ,c

        imgs = [np.moveaxis(img[:, i * w: (i + 1) * w], -1, 0) for i in range(LEN_IMG)]

        input_images.append(imgs[0])
        target_images.append(imgs[2])
        generated_images.append(imgs[IDX_FAKE])

        assert img_name.endswith('_vis.png') or img_name.endswith(
            '_vis.jpg'), 'unexpected img name: should end with _vis.png'
        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        fr = img_name[0]
        to = img_name[1]

        names.append([fr, to])

    return input_images, target_images, generated_images, names


def evaluate_is(generated_images_dir):
    input_images, target_images, generated_images, names = load_generated_images(generated_images_dir)

    IS, IS_std = get_inception_score(torch.FloatTensor(generated_images))
    print(IS, IS_std)


if __name__ == "__main__":
    LEN_IMG = 5
    IDX_FAKE = 4

    generated_images_dir = './results/market_APS/test_latest/images'
    annotations_file_test = './dataset/market_data/market-annotation-test.csv'

    args = sys.argv[1:].copy()
    if len(args):
        generated_images_dir = f'./results/{args[0]}/test_latest/images'
    if len(args) > 1:
        LEN_IMG = int(args[1])
    if len(args) > 2:
        IDX_FAKE = int(args[2])

    evaluate_is(generated_images_dir)
