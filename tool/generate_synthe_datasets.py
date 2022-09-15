import numpy as np
import pandas as pd
import json
import os
import sys
import random
import glob
from PIL import Image

from generate_dataset_utils import compute_pose

random.seed()

MISSING_VALUE = -1

SYNTHE_PATH = r'D:\Mingming\synthe_dripe\output_temp'
IMG_DIR = f'{SYNTHE_PATH}/images'

DEBUG = False


# Coco to CSV

def make_pairs(imgs_lst):
    imgs_splt = [img.split('_') for img in imgs_lst]
    configs = {
        '_'.join(img_k[:2]):
            ['_'.join(img) for img in imgs_splt if '_'.join(img_k[:2]) == '_'.join(img[:2])] for img_k in imgs_splt
    }

    pairs = []
    for imgs_conf in configs.values():
        imgs_conf_2 = imgs_conf.copy()
        random.shuffle(imgs_conf_2)
        pairs += [(imgs_conf[k], imgs_conf_2[k]) for k in range(len(imgs_conf))]

    return pairs


def blender_to_csv():
    csv_annots = []

    with open(os.path.join(SYNTHE_PATH, 'infos.json')) as f_info:
        infos = json.load(f_info)

    annotation_files = glob.glob(os.path.join(SYNTHE_PATH, 'annots_2D', '*.csv'))
    csv_in_blender = ['nose', 'neck_01_head', 'upperarm_r_head', 'lowerarm_r_head',
                      'hand_r_head', 'upperarm_l_head', 'lowerarm_l_head', 'hand_l_head',
                      'thigh_r_head', 'calf_r_head', 'calf_r_tail', 'thigh_l_head',
                      'calf_l_head', 'calf_l_tail', 'eye_r', 'eye_l', 'ear_r', 'ear_l']
    img_size = None
    for ann_file in annotation_files:
        annotations = pd.read_csv(ann_file, sep=';')
        img_root = '_'.join(ann_file.split('_')[-2:])[:-4]

        for i in range(annotations.shape[0]):
            annot = annotations.iloc[[i]]
            xs = [int(annot[lbl + '_x']) for lbl in csv_in_blender]
            ys = [int(annot[lbl + '_y']) for lbl in csv_in_blender]

            # Thorax
            xs[1] = round((float(annot['clavicle_l_head_x']) + float(annot['clavicle_r_head_x'])) / 2.)
            ys[1] = round((float(annot['clavicle_l_head_y']) + float(annot['clavicle_r_head_y'])) / 2.)

            candidates_img = glob.glob(os.path.join(IMG_DIR, img_root + f'_{i}.png')) + \
                             glob.glob(os.path.join(IMG_DIR, img_root + f'_{i}_*.png'))
            assert len(candidates_img) == 1
            img_path = candidates_img[0]
            if img_size is None:
                with Image.open(img_path) as img:
                    img_size = img.size
            img_name = os.path.basename(img_path)

            for k in range(len(xs)):
                if xs[k] >= img_size[0] or xs[k] < 0 or ys[k] >= img_size[1] or ys[k] < 0:
                    xs[k] = MISSING_VALUE
                    ys[k] = MISSING_VALUE

            csv_annots.append((
                img_name,
                ys,
                xs,
            ))

    pairs = make_pairs([line[0] for line in csv_annots])
    return csv_annots, pairs


def generate_set(set_name):
    print('\nComputing pose')
    annotations_file = f'{SYNTHE_PATH}/synthe-annotation-{set_name}.csv'  # pose annotation path
    save_path = f'{SYNTHE_PATH}/{set_name}K'  # path to store pose maps
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    compute_pose(annotations_file, save_path, img_dir=IMG_DIR, debug=DEBUG)


def generate_dataset():
    annotations, pairs = blender_to_csv()
    test_ratio = 0.05
    models_count = {}
    for img in annotations:
        model = img[0].split('_')[0]
        models_count[model] = models_count.setdefault(model, 0) + 1

    models = list(models_count.keys())
    random.shuffle(models)
    nb_imgs = sum(models_count.values())
    for set_name, ratio in {'test': 0.05, 'train': 1 - .05}.items():
        annotations_set = []
        pairs_set = []
        while models and len(annotations_set) < nb_imgs * ratio:
            model = models.pop()
            annotations_set += [anno for anno in annotations if anno[0].split('_')[0] == model]
            pairs_set += [pair for pair in pairs if pair[0].split('_')[0] == model]

        if not DEBUG:
            with open(os.path.join(SYNTHE_PATH, f'synthe-pairs-{set_name}.csv'), 'w') as f:
                f.write('from,to\n' + '\n'.join(','.join(p) for p in pairs_set))
            with open(os.path.join(SYNTHE_PATH, f'synthe-annotation-{set_name}.csv'), 'w') as f:
                f.write(
                    'name:keypoints_y:keypoints_x\n' + '\n'.join(
                        ': '.join(str(a) for a in an) for an in annotations_set))

            generate_set(set_name)


if __name__ == '__main__':
    generate_dataset()
