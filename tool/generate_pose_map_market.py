import numpy as np
import pandas as pd
import json
import os

from generate_dataset_utils import cords_to_map, load_pose_cords_from_strings

MISSING_VALUE = -1


def compute_pose(image_dir, annotations_file, savePath):
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')
    image_size = (128, 64)
    cnt = len(annotations_file)
    for i in range(cnt):
        print('processing %d / %d ...' % (i, cnt))
        row = annotations_file.iloc[i]
        name = row.name
        print(savePath, name)
        file_name = os.path.join(savePath, name + '.npy')
        kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        pose = cords_to_map(kp_array, image_size)
        print(np.sum(pose))
        np.save(file_name, pose)


# PATH train
img_dir = './dataset/market_data/train'  # raw image path
annotations_file = './dataset/market_data/market-annotation-train.csv'  # pose annotation path
save_path = './dataset/market_data/trainK'  # path to store pose maps
compute_pose(img_dir, annotations_file, save_path)

# PATH test
img_dir = './dataset/market_data/test'  # raw image path
annotations_file = './dataset/market_data/market-annotation-test.csv'  # pose annotation path
save_path = './dataset/market_data/testK'  # path to store pose maps
compute_pose(img_dir, annotations_file, save_path)
