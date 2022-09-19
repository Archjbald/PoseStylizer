import json
import os

import numpy as np
import pandas as pd
from PIL import Image

MISSING_VALUE = -1


def resize_img(img, crop_ratio, crop):
    img_croped = img.crop(crop)
    img_resized = img_croped.resize((round(img_croped.width * crop_ratio), round(img_croped.height * crop_ratio)))

    return img_resized


def crop_img(img, crop_ratio, crop):
    img_resized = img.resize((round(img.width * crop_ratio), round(img.height * crop_ratio)))
    img_croped = img_resized.crop((
        int(crop_ratio * crop[0]),
        int(crop_ratio * crop[1]),
        int(crop_ratio * crop[0]) + 256,
        int(crop_ratio * crop[1]) + 192,
    ))
    return img_croped


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float16')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result


def compute_pose(annotations_file, savePath, img_dir, debug=False):
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')
    cnt = len(annotations_file)
    for i in range(cnt):
        print('processing %d / %d ...' % (i, cnt))
        row = annotations_file.iloc[i]
        name = row.name
        img_path = os.path.join(img_dir, name)
        file_name = os.path.join(savePath, name + '.npy')
        kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        img = Image.open(img_path)
        pose = cords_to_map(kp_array, img.size[::-1], sigma=12)
        if not debug:
            np.save(file_name, pose)
