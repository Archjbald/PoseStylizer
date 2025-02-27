import os
from inception_score import get_inception_score

from skimage.io import imread, imsave
from skimage.metrics import structural_similarity

import numpy as np
import pandas as pd


def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image / 255.0 - 0.5) - 2 * (generated_image / 255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list), np.std(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = structural_similarity(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False, multichannel=True,
                                     data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list), np.std(ssim_score_list)


def save_images(input_images, target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for images in zip(input_images, target_images, generated_images, names):
        res_name = str('_'.join(images[-1])) + '.png'
        imsave(os.path.join(output_folder, res_name), np.concatenate(images[:-1], axis=1))


def create_masked_image(names, images, annotation_file):
    import pose_utils
    masked_images = []
    df = pd.read_csv(annotation_file, sep=':')
    for name, image in zip(names, images):
        to = name[1]
        ano_to = df[df['name'] == to].iloc[0]

        kp_to = pose_utils.load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = pose_utils.produce_ma_mask(kp_to, image.shape[:2])
        masked_images.append(image * mask[..., np.newaxis])

    return masked_images


def addBounding(image, bound=40):
    h, w, c = image.shape
    image_bound = np.ones((h, w + bound * 2, c)) * 255
    image_bound = image_bound.astype(np.uint8)
    image_bound[:, bound:bound + w] = image

    return image_bound


def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []

    names = []
    for img_name in os.listdir(images_folder):
        img = imread(os.path.join(images_folder, img_name))
        w = int(img.shape[1] / 5)  # h, w ,c

        input_images.append(addBounding(img[:, :w]))
        target_images.append(addBounding(img[:, 2 * w:3 * w]))
        generated_images.append(addBounding(img[:, 4 * w:5 * w]))

        assert img_name.endswith('_vis.png') or img_name.endswith(
            '_vis.jpg'), 'unexpected img name: should end with _vis.png'
        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        fr = img_name[0]
        to = img_name[1]

        names.append([fr, to])

    return input_images, target_images, generated_images, names


def test(generated_images_dir, annotations_file_test):
    # load images
    print("Loading images...")
    input_images, target_images, generated_images, names = load_generated_images(generated_images_dir)

    print("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print("SSIM score ", structured_score)
    print(generated_images_dir)

    print("Compute l1 score...")
    norm_score = l1_score(generated_images, target_images)
    print("L1 score ", norm_score)

    print("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print("Inception score ", inception_score)
    print("Inception score ", inception_score, " SSIM score ", structured_score, " L1 score ", norm_score)


if __name__ == "__main__":
    # fix these paths
    generated_images_dir = './results/fashion_APS/test_latest/images'
    annotations_file_test = './dataset/fashion_data/fasion-resize-annotation-test.csv'

    test(generated_images_dir, annotations_file_test)
    print(generated_images_dir)
