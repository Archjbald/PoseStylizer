import os
import json
from collections import OrderedDict

import pandas as pd
import numpy as np
from PIL import Image

MISSING_VALUE = -1


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result


def get_middle(x, y, k_1, k_2):
    if x[k_1] != MISSING_VALUE and x[k_2] != MISSING_VALUE and y[k_1] != MISSING_VALUE and y[k_2] != MISSING_VALUE:
        a = round((x[k_1] + x[k_2]) / 2.)
        b = round((y[k_1] + y[k_2]) / 2.)
        return a, b

    return MISSING_VALUE, MISSING_VALUE


def compute_additional_kps(kps_conf, annots):
    """

    :param annots: base annotations
    :param kps_conf: {id: (id_1, id_2)}
    :return: new annots
    """

    annots = np.concatenate((annots, np.full((max(kps_conf) + 1 - len(annots), 2), MISSING_VALUE)))

    x, y = annots[:, 1], annots[:, 0]
    print(annots)
    for k, (k_1, k_2) in kps_conf.items():
        if not isinstance(k_1, int):
            a_1, b_1 = get_middle(x, y, *k_1)
            a_2, b_2 = get_middle(x, y, *k_2)
            if not a_1 == MISSING_VALUE and not a_2 == MISSING_VALUE:
                x[k] = round((a_1 + a_2) / 2.)
                y[k] = round((b_1 + b_2) / 2.)
            elif not a_1 == MISSING_VALUE:
                x[k], y[k] = a_1, b_1
            elif not a_2 == MISSING_VALUE:
                x[k], y[k] = a_2, b_2
        else:
            x[k], y[k] = get_middle(x, y, k_1, k_2)

    print(annots)
    return annots


def extend_dataset(img_dir, annotations_path, save_path):
    annotations_file = pd.read_csv(annotations_path, sep=':')
    annotations_file = annotations_file.set_index('name')
    cnt = len(annotations_file)

    # compute thorax
    kps_conf = {
        1: (2, 5),  # thorax
        18: ((5, 8), (2, 11)),  # tommy
        19: (2, 3),  # upper arm R
        20: (3, 4),  # lower arm R
        21: (5, 6),  # upper arm L
        22: (6, 7),  # lower arm L
        23: (8, 9),  # thigh R
        24: (9, 10),  # calf R
        25: (11, 12),  # thigh L
        26: (12, 13),  # calf L

    }
    new_annotations = OrderedDict()
    for i in range(cnt):
        print('processing %d / %d ...' % (i, cnt))
        row = annotations_file.iloc[i]
        name = row.name
        print(save_path, name)
        img_path = os.path.join(img_dir, name)
        file_name = os.path.join(save_path, name + '.npy')
        kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        kp_array = compute_additional_kps(kps_conf, kp_array)
        img = Image.open(img_path)
        pose = cords_to_map(kp_array, img.size[::-1], sigma=12)
        np.save(file_name, pose)
        new_annotations[name] = kp_array

    text_annots = ['name:keypoints_y:keypoints_x']
    for name, anno in new_annotations.items():
        anno = anno.transpose().tolist()
        text_annots.append(':'.join([name] + [str(an) for an in anno]))

    with open(annotations_path.replace('.csv', '_extand.csv'), 'w') as f:
        f.write('\n'.join(text_annots))
    return None


if __name__ == '__main__':
    # PATH train
    img_dir = './dataset/fashion_data/train'  # raw image path
    annotations_path = './dataset/fashion_data/fashion-resize-annotation-train.csv'  # pose annotation path
    save_path = './dataset/fashion_data/trainK'  # path to store pose maps
    extend_dataset(img_dir, annotations_path, save_path)

    # PATH test
    img_dir = './dataset/fashion_data/test'  # raw image path
    annotations_path = './dataset/fashion_data/fashion-resize-annotation-test.csv'  # pose annotation path
    save_path = './dataset/fashion_data/testK'  # path to store pose maps
    extend_dataset(img_dir, annotations_path, save_path)
