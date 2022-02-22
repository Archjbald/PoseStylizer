import os
import shutil
import pandas as pd
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    new_root = './fashion_data'
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    train_root = './fashion_data/train'
    if not os.path.exists(train_root):
        os.mkdir(train_root)

    test_root = './fashion_data/test'
    if not os.path.exists(test_root):
        os.mkdir(test_root)

    train_images = []
    train_f = open('./fashion_data/train.lst', 'r')
    for lines in train_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            train_images.append(lines)

    test_images = []
    test_f = open('./fashion_data/test.lst', 'r')
    for lines in test_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)

    print(train_images, test_images)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                path_names = path.split('/')
                # path_names[2] = path_names[2].replace('_', '')
                path_names[3] = path_names[3].replace('_', '')
                path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
                path_names = "".join(path_names)
                # new_path = os.path.join(root, path_names)
                img = Image.open(path)
                imgcrop = img.crop((40, 0, 216, 256))
                if new_path in train_images:
                    imgcrop.save(os.path.join(train_root, path_names))
                elif new_path in test_images:
                    imgcrop.save(os.path.join(test_root, path_names))

def compute_additional_kps(kps_conf, annots):
    """

    :param annots: base annotations
    :param kps_conf: {id: (id_1, id_2)}
    :return: new annots
    """

    for an in annots:
        an += [MISSING_VALUE for _ in range(max(kps_conf) + 1 - len(an))]

    x, y = annots

    for k, (k_1, k_2) in kps_conf.items():
        if x[k_1] != MISSING_VALUE and x[k_2] != MISSING_VALUE and y[k_1] != MISSING_VALUE and y[k_2] != MISSING_VALUE:
            x[k] = round((x[k_1] + x[k_2]) / 2.)
            y[k] = round((y[k_1] + y[k_2]) / 2.)

    return x, y


def regenerate_set(set_name):
    annotations_file = f'./dataset/fashion_data/{set_name}K'
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
        np.save(file_name, pose)


make_dataset('./fashion')
