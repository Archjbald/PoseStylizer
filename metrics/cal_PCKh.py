import math
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm  # progress bar
import torch
import torchvision.transforms as trans

from util.util import get_kps
from metrics.utils import resize_keypoints

MISSING_VALUE = -1

PARTS_SEL = [0, 1, 14, 15, 16, 17]
'''
  hz: head size
  alpha: norm factor
  px, py: predict coords
  tx, ty: target coords
'''

ANNOTS_TARGETS = 'annots_real.csv'
ANNOTS_PREDS = 'annots_fake.csv'


def isRight(px, py, tx, ty, hz, alpha):
    if px == -1 or py == -1 or tx == -1 or ty == -1:
        return 0

    if abs(px - tx) < hz[0] * alpha and abs(py - ty) < hz[1] * alpha:
        return 1
    else:
        return 0


def how_many_right_seq(px, py, tx, ty, hz, alpha):
    nRight = 0
    for i in range(len(px)):
        nRight = nRight + isRight(px[i], py[i], tx[i], ty[i], hz, alpha)

    return nRight


def valid_points(tx):
    nValid = 0
    for item in tx:
        if item != -1:
            nValid = nValid + 1
    return nValid


def get_head_wh(x_coords, y_coords):
    final_w, final_h = -1, -1
    component_count = 0
    save_componets = []
    for component in PARTS_SEL:
        if x_coords[component] == MISSING_VALUE or y_coords[component] == MISSING_VALUE:
            continue
        else:
            component_count += 1
            save_componets.append([x_coords[component], y_coords[component]])
    if component_count >= 2:
        x_cords = []
        y_cords = []
        for component in save_componets:
            x_cords.append(component[0])
            y_cords.append(component[1])
        xmin = min(x_cords)
        xmax = max(x_cords)
        ymin = min(y_cords)
        ymax = max(y_cords)
        final_w = xmax - xmin
        final_h = ymax - ymin

    return final_w, final_h


def get_head_size(x_coords, y_coords):
    return [round(max(get_head_wh(x_coords, y_coords)) / 0.7), ] * 2


def get_pckh_from_dir(results_dir):
    target_annotation = os.path.join(results_dir, ANNOTS_TARGETS)
    pred_annotation = os.path.join(results_dir, ANNOTS_PREDS)
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


def get_pckh_from_hpe(img_loader, hpe_net, target_annotation, gt_size):
    # target_annotation = os.path.join(results_dir, ANNOTS_TARGETS)

    if not isinstance(target_annotation, pd.DataFrame):
        tAnno = pd.read_csv(target_annotation, sep=':')
    else:
        tAnno = target_annotation

    nAll = 0
    nCorrect = 0
    alpha = 0.5
    nbImg = 0
    for i, img in enumerate(tqdm(img_loader)):
        hmaps = hpe_net(img.cuda() if torch.cuda.is_available() else img)[0]
        kps, vis = get_kps(hmaps)
        # hmaps_sized = trans.functional.resize(hmaps, gt_size)
        # kps, vis = get_kps(hmaps_sized)

        kps[vis == False] = MISSING_VALUE
        img_name = img_loader.dataset.get_name(i)

        pycords, pxcords = kps[0].t().tolist()

        tValues = tAnno.query('name == "%s"' % (img_name))
        if not len(tValues):
            continue
        else:
            tValues = tValues.values[0]

        tycords = json.loads(tValues[1])
        # tycords = [v if k not in [5, 6, 7, 11, 12, 13, 15, 17] else MISSING_VALUE for k, v in
        #            enumerate(tycords)]  # list of numbers
        txcords = json.loads(tValues[2])

        txcords_rs, tycords_rs = resize_keypoints(txcords, tycords, gt_size, list(img.shape[-2:]))

        xBox, yBox = get_head_size(txcords_rs, tycords_rs)
        if xBox == -1 or yBox == -1:
            continue

        head_size = (xBox, yBox)
        nAll = nAll + valid_points(tycords)
        nCorrect = nCorrect + how_many_right_seq(pxcords, pycords, txcords_rs, tycords_rs, head_size, alpha)
        nbImg += 1

    pckh = nCorrect * 1.0 / nAll
    print(f'PCKh calculated for {nbImg} images.')
    # print(f'{nCorrect}/{nAll} : {pckh:.3f}%')

    return pckh, nCorrect, nAll


if __name__ == '__main__':
    ANNOTS_PREDS.replace('.csv', '_temp.csv')
    ANNOTS_TARGETS.replace('.csv', '_temp.csv')
    import torch
