import glob
import os

import numpy as np
from PIL import Image

def get_square(img_0):
    img_square = np.zeros((256, 256), dtype=np.uint8)
    height = img_0.shape[0]
    width = img_0.shape[1]
    x_start = (256 - height) // 2
    y_start = (256 - width) // 2
    img_square[x_start:x_start + height, y_start:y_start + width] = img_0
    img_square = Image.fromarray(img_square)
    return img_square


def reorder_lbls(img_0):
    img_labeled = img_0.copy()
    lbls = [0, 1, 1, 0, 0, 2, 3, 2, 7, 3, 6, 0, 3, 5, 6, 6, 7, 7, 4, 4, 0]
    for i, lbl in enumerate(lbls):
        img_labeled[img_0 == i] = lbl

    return img_labeled


def prepare_spl(i_path):
    img_0 = np.array(Image.open(i_path))
    img_0 = reorder_lbls(img_0)
    img_square = get_square(img_0)
    img_square.save(i_path.replace('cihp_parsing_maps', "SPL8"))


if __name__ == '__main__':
    for i_path in glob.glob(r"D:\Datasets\DrAIver\cihp_parsing_maps\*.png"):
        prepare_spl(i_path)
