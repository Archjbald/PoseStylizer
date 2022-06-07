import os
import sys
import json
import glob

import skimage
import numpy as np
import pandas as pd
from PIL import Image

from evaluate_IS import get_inception_score
from evaluate_FID import get_fid

from calPCKH_market import get_head_wh, valid_points, how_many_right_seq


def load_generated_images(images_folder, len_img, idx_fake):
    input_images = []
    generated_images = []

    names = []
    img_list = os.listdir(images_folder)
    for img_name in img_list:
        img = skimage.io.imread(os.path.join(images_folder, img_name))
        w = int(img.shape[1] / len_img)  # h, w ,c

        imgs = [img[:, i * w: (i + 1) * w] for i in range(len_img)]

        input_images += [imgs[0], imgs[3]]
        generated_images += [imgs[idx_fake], imgs[idx_fake + 1]]

        assert img_name.endswith('_vis.png') or img_name.endswith(
            '_vis.jpg'), 'unexpected img name: should end with _vis.png'
        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        fr = img_name[0]
        to = img_name[1]

        names.append([fr, to])

    Image.fromarray(generated_images[0]).save(os.path.join(images_folder, '../sample_output.png'))

    return (np.stack(input_images, axis=0), np.stack(generated_images, axis=0), names)


def get_pckh(results_dir):
    target_annotation = os.path.join(results_dir, 'annots_real.csv')
    pred_annotation = os.path.join(results_dir, 'annots_fake.csv')
    tAnno = pd.read_csv(target_annotation, sep=':')
    pAnno = pd.read_csv(pred_annotation, sep=':')

    pRows = pAnno.shape[0]

    nAll = 0
    nCorrect = 0
    alpha = 0.5
    for i in range(pRows):
        pValues = pAnno.iloc[i].values
        pname = pValues[0]
        pycords = json.loads(pValues[1])  # list of numbers
        pxcords = json.loads(pValues[2])

        tValues = tAnno.query('name == "%s"' % (pname)).values[0]
        tycords = json.loads(tValues[1])  # list of numbers
        txcords = json.loads(tValues[2])

        xBox, yBox = get_head_wh(txcords, tycords)
        if xBox == -1 or yBox == -1:
            continue

        head_size = (xBox, yBox)
        nAll = nAll + valid_points(tycords)
        nCorrect = nCorrect + how_many_right_seq(pxcords, pycords, txcords, tycords, head_size, alpha)

    pckh = nCorrect * 1.0 / nAll
    # print(f'{nCorrect}/{nAll} : {pckh:.3f}%')

    return pckh, nCorrect, nAll


def get_metrics(results_dir, len_img, idx_fake):
    print('Loading images from ', results_dir)
    input_images, generated_images, names = \
        load_generated_images(os.path.join(results_dir, 'images'), len_img, idx_fake)
    print(f'{len(input_images)} images loaded\n')


    print('Input images...')
    IS_input = get_inception_score(input_images)
    print(f"IS input: {IS_input[0]}, std: {IS_input[1]}")

    print('Input generated....')
    IS_output = get_inception_score(generated_images)
    print(f"IS output: {IS_output[0]}, std: {IS_output[1]}")

    print('FID...')
    FID = get_fid(input_images, generated_images)
    print("FID: ", FID)

    PCKs = get_pckh(results_dir)
    print(f'PCKh: {PCKs[0] * 100:.2f}% ({PCKs[1]}/{PCKs[2]} )')


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
    len_img = 6
    idx_fake = 4

    results_dir = './results/market_APS'
    annotations_file_test = './dataset/market_data/market-annotation-test.csv'

    args = sys.argv[1:].copy()
    if len(args):
        results_dir = f'./results/{args[0]}'
    if len(args) > 1:
        len_img = int(args[1])
    if len(args) > 2:
        idx_fake = int(args[2])

    results_dir = get_last_dir(results_dir)

    return results_dir, len_img, idx_fake


if __name__ == '__main__':
    get_metrics(*get_args())
