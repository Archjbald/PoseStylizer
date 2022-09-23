import numpy as np
import pandas as pd
import json
import os

from skimage.draw import disk
import skimage.measure
import skimage.transform

from util.util import get_kps

LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
          'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1


def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
    yv, xv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                         sparse=False, indexing='xy')

    a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
    b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
    c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

    return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))


def make_gaussian_limb_masks(BP):
    limbs = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]
    # limbs = [[0, 1, 14, 15, 16, 17], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    n_limbs = len(limbs)
    img_height, img_width = BP.shape[1], BP.shape[2]
    mask = np.zeros((img_height, img_width, n_limbs))

    joints, vis = get_kps(BP)
    joints[~vis] = MISSING_VALUE

    # Gaussian sigma perpendicular to the limb axis.
    img_diag = (img_width ** 2 + img_height ** 2) ** 0.5
    sigma_perp = np.array([0.04 * img_diag] * 9 + [0.055 * img_diag]) ** 2
    # sigma_perp = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 13]) ** 2
    # sigma_perp = np.array([9, 9, 9, 9, 9, 9, 9, 9, 9, 13]) ** 2

    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            missing = joints[limbs[i][j]][0] == MISSING_VALUE or joints[limbs[i][j]][1] == MISSING_VALUE
            if missing:
                break
            p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]
        if missing:
            continue
        if n_joints_for_limb == 4:
            p_top = np.mean(p[0:2, :], axis=0)
            p_bot = np.mean(p[2:4, :], axis=0)
            p = np.vstack((p_top, p_bot))

        center = np.mean(p, axis=0)

        sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.2])
        theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

        mask_i = make_gaussian_map(img_width, img_height, center, sigma_parallel, sigma_perp[i], theta)
        mask[:, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

    bg_mask = np.expand_dims(1.0 - np.amax(mask, axis=2), 2)
    mask = np.concatenate((bg_mask, mask), axis=2)
    mask = mask.transpose(-1, 0, -2)  # h,w,c --> c,h,w

    return mask
