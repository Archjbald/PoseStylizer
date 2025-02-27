import numpy as np
import pandas as pd
import json
import os
import sys
import random
import glob
from PIL import Image

MISSING_VALUE = -1

DRIPE_IMGS = r'D:\Datasets\Autob_20k\set_pics'

DRIPE_PATH = './dataset/dripe_data'
IMG_DIR = f'{DRIPE_PATH}/images'

CROP = False

DEBUG = True


# Coco to CSV

def make_pairs(images):
    imgs_lst = list(images.values())

    imgs_splt = [img.replace('-', '_').split('_') for img in imgs_lst]
    imgs_splt = [['-'.join(img[:-3])] + img[-3:] for img in imgs_splt]

    imgs_dict = {}
    for img in imgs_splt:
        imgs_dict.setdefault(img[0], {}).setdefault(img[1], []).append('_'.join(img[2:]))

    pairs = []
    for act in imgs_dict:
        vids = list(imgs_dict[act].keys())
        while len(vids) > 1:
            vid1, vid2 = random.sample(vids, 2)

            frame1 = random.choice(imgs_dict[act][vid1])
            frame2 = random.choice(imgs_dict[act][vid2])

            pairs.append((
                f'{act}-{vid1}_{frame1}',
                f'{act}-{vid2}_{frame2}',
            ))

            vids.remove(vid1)
            vids.remove(vid2)

    return pairs


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


def compute_crop(bbox, img_size):
    crop_size = (256., 192.)
    ratio_x = crop_size[0] / bbox[2]
    ratio_y = crop_size[1] / bbox[3]

    if ratio_x < ratio_y:
        if crop_size[1] / ratio_x > img_size[1]:
            crop_ratio = ratio_y
        else:
            crop_ratio = ratio_x
    else:
        if crop_size[0] / ratio_y > img_size[0]:
            crop_ratio = ratio_x
        else:
            crop_ratio = ratio_y

    crop_w = min(round(crop_size[0] / crop_ratio), img_size[0])
    crop_h = min(round(crop_size[1] / crop_ratio), img_size[1])

    assert round(crop_w * crop_ratio) == crop_size[0]
    assert round(crop_h * crop_ratio) == crop_size[1]

    bbox_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
    crop = [
        bbox_center[0] - crop_w // 2,
        bbox_center[1] - crop_h // 2,
        -1,
        -1,
    ]
    crop[2] = crop[0] + crop_w
    crop[3] = crop[1] + crop_h

    assert crop[2] - crop[0] <= img_size[0] and crop[3] - crop[1] <= img_size[1]

    if crop[0] < 0:
        crop[2] = crop_w
        crop[0] = 0
    elif crop[2] >= img_size[0]:
        crop[0] = img_size[0] - crop_w
        crop[2] = img_size[0]
    elif crop[1] < 0:
        crop[3] = crop_h
        crop[1] = 0
    elif crop[3] >= img_size[1]:
        crop[1] = img_size[1] - crop_h
        crop[3] = img_size[1]

    return crop_ratio, crop


def convert_annot(annot, crop_ratio, crop_x, crop_y):
    kps = annot['keypoints']

    x_coco = []
    y_coco = []
    for i in range(17):
        vis = kps[3 * i + 2] > 0
        x_coco.append(int((kps[3 * i] - crop_x) * crop_ratio) if vis else MISSING_VALUE)
        y_coco.append(int((kps[3 * i + 1] - crop_y) * crop_ratio) if vis else MISSING_VALUE)

    # csv_in_coco[kp in csv] = kp in coco
    csv_in_coco = [0, 0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    x = [x_coco[kp_csv] for kp_csv in csv_in_coco]
    y = [y_coco[kp_csv] for kp_csv in csv_in_coco]
    # compute thorax
    if x[2] != MISSING_VALUE and x[5] != MISSING_VALUE and y[2] != MISSING_VALUE and y[5] != MISSING_VALUE:
        x[1] = round((x[2] + x[5]) / 2.)
        y[1] = round((y[2] + y[5]) / 2.)

    return x, y


def coco_to_csv(coco_json, set_name):
    coco_annots = coco_json['annotations']
    coco_images = coco_json['images']

    images = dict([(im['id'], im['file_name']) for im in coco_images])

    csv_annots = []
    for i, annot in enumerate(coco_annots):
        print('processing %d / %d ...' % (i, len(coco_annots)))

        image_id = annot['image_id']
        # filter out non-head images
        if annot['keypoints'][3 * 4 + 1] < 60:
            del images[annot['image_id']]
            continue

        img = Image.open(os.path.join(DRIPE_IMGS, images[image_id]))
        crop_ratio, crop_box = compute_crop(annot['bbox'], img.size)
        x, y = convert_annot(annot, crop_ratio, crop_box[0], crop_box[1])
        csv_annots.append((
            images[image_id],
            y,
            x,
        ))
        if CROP:
            img = crop_img(img, crop_ratio, crop_box)
            if not DEBUG:
                img.save(os.path.join(IMG_DIR, images[image_id]))

    pairs = make_pairs(images)

    if not DEBUG:
        with open(os.path.join(DRIPE_PATH, f'dripe-pairs-{set_name}.csv'), 'w') as f:
            f.write('from,to\n' + '\n'.join(','.join(p) for p in pairs))
        with open(os.path.join(DRIPE_PATH, f'dripe-annotation-{set_name}.csv'), 'w') as f:
            f.write('name:keypoints_y:keypoints_x\n' + '\n'.join(': '.join(str(a) for a in an) for an in csv_annots))


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


def compute_pose(annotations_file, savePath):
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')
    cnt = len(annotations_file)
    for i in range(cnt):
        print('processing %d / %d ...' % (i, cnt))
        row = annotations_file.iloc[i]
        name = row.name
        img_path = os.path.join(IMG_DIR, name)
        file_name = os.path.join(savePath, name + '.npy')
        kp_array = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        img = Image.open(img_path)
        pose = cords_to_map(kp_array, img.size[::-1], sigma=12)
        if not DEBUG:
            np.save(file_name, pose)


def generate_set(set_name):
    js_path = f'D:/Datasets/Autob_20k/splits/split_1/autob_coco_{set_name}.json'
    with open(js_path) as f:
        coco = json.load(f)
    print('Converting COCO')
    coco_to_csv(coco, set_name)

    print('\nComputing pose')
    annotations_file = f'{DRIPE_PATH}/dripe-annotation-{set_name}.csv'  # pose annotation path
    save_path = f'{DRIPE_PATH}/{set_name}K'  # path to store pose maps
    compute_pose(annotations_file, save_path)


if __name__ == '__main__':
    for set_name in ['test']:
        generate_set(set_name)
