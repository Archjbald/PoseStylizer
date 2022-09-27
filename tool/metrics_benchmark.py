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


def get_len_img(img):
    if img.ndim > 2:
        img = img.max(axis=img.shape.index(3))
    rows = img.mean(axis=0) > 80
    buff = 0
    state = rows[0]
    count = [0]
    for r in rows:
        if r == state:
            count[-1] += 1 + buff
            buff = 0
            continue
        buff += 1
        if buff > 10:
            state = r
            count.append(buff)
            buff = 0

    count.sort()
    width = img.shape[1]
    for c in count[-2:0:-1]:
        if not width % c:
            return width // c

    return None


def load_generated_images(images_folder, idx_fake):
    input_images = []
    generated_images = []

    len_img = None

    names = []
    img_list = os.listdir(images_folder)
    for k, img_name in enumerate(img_list):

        img = skimage.io.imread(os.path.join(images_folder, img_name))
        if not k:
            if 'fashion' in images_folder:
                ratio = 1.45
            elif 'market' in images_folder:
                ratio = 2.
            elif 'synthe' in images_folder:
                ratio = 0.75

            len_img = round(img.shape[1] / (img.shape[0] / ratio))
            print('Len_img set to: ', len_img)

        w = int(img.shape[1] / len_img)  # h, w ,c

        imgs = [img[:, i * w: (i + 1) * w] for i in range(len_img)]

        assert img_name.endswith('_vis.png') or img_name.endswith(
            '_vis.jpg'), 'unexpected img name: should end with _vis.png'
        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'

        for s in (0, 1):
            name = img_name[s]
            # if name in names:
            #     continue
            names.append(name)
            input_images.append(imgs[2 * s])

        generated_images += [imgs[idx_fake]]

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
        load_generated_images(os.path.join(results_dir, 'images'), idx_fake)
    print(f'{len(input_images)} images loaded\n')

    # get_detection_score(input_images)

    source_images = input_images[0::2]
    target_images = input_images[1::2]

    print('Input images...')
    IS_input = get_inception_score(source_images)
    print(f"IS input: {IS_input[0]}, std: {IS_input[1]}")

    print('Input generated....')
    IS_output = get_inception_score(generated_images)
    print(f"IS output: {IS_output[0]}, std: {IS_output[1]}")

    print('FID...')
    FID = get_fid(source_images, generated_images)
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
