import os
import time

import numpy as np
import torch
from PIL import Image
import skimage


def get_len_img(img):
    if img.ndim > 2:
        img = img.max(axis=img.shape.index(3))
    rows = img.mean(axis=0) > 80
    buff = 0
    state = rows[0]
    count = [0]
    for r in rows:
        if r == state:
            count[-1] += 1 + buff
            buff = 0
            continue
        buff += 1
        if buff > 10:
            state = r
            count.append(buff)
            buff = 0

    count.sort()
    width = img.shape[1]
    for c in count[-2:0:-1]:
        if not width % c:
            return width // c

    return None


def load_generated_images(images_folder, idx_fake):
    input_images = []
    generated_images = []

    len_img = None

    names = []
    img_list = os.listdir(images_folder)
    for k, img_name in enumerate(img_list):
        # if k > 100:
        #     break

        img = skimage.io.imread(os.path.join(images_folder, img_name))
        if not k:
            len_img = get_len_img(img)
        w = int(img.shape[1] / len_img)  # h, w ,c

        imgs = [img[:, i * w: (i + 1) * w] for i in range(len_img)]

        assert img_name.endswith('_vis.png') or img_name.endswith(
            '_vis.jpg'), 'unexpected img name: should end with _vis.png'
        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'

        # fr = img_name[0]
        # to = img_name[1]
        #
        # names.append([fr, to])

        for s in (0, 1):
            name = img_name[s]
            if name in names:
                continue
            names.append(name)
            input_images.append(imgs[2 * s])

        # input_images += [imgs[0], imgs[2]]
        generated_images += [imgs[idx_fake]]

    Image.fromarray(generated_images[0]).save(os.path.join(images_folder, '../sample_output.png'))

    return (np.stack(input_images, axis=0), np.stack(generated_images, axis=0), names)


def get_person_confidence(results):
    person_idx = results.names.index('person')
    preds = results.pred
    max_confidence = []
    for i, pred in enumerate(preds):
        if not pred.numel():
            max_confidence.append(0)
        else:
            person_preds = pred[pred[:, -1] == person_idx]
            if not person_preds.numel():
                max_confidence.append(0)
            else:
                max_confidence.append(float(person_preds.max(dim=0).values[-2]))

    return max_confidence


def get_detection_score(images, batch_size=10):
    if images.shape[1] == 3:
        images = np.moveaxis(np.array(images), 1, -1)

    assert (type(images) == np.ndarray)
    assert (len(images.shape) == 4)
    assert (images.shape[-1] == 3)
    assert (np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'

    print('Calculating Detection Score with %i images in batches of %i' % (images.shape[0], batch_size))
    start_time = time.time()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    scores = []
    for i in range(round(len(images) / batch_size)):
        batch = list(images[i * batch_size: (i + 1) * batch_size])
        results = model(batch)
        scores += get_person_confidence(results)

    total_ds = (sum(scores) / len(scores)) if scores else 0
    print(f'Detection score: {total_ds * 100:.1f}%')
    print('\nDetection Score calculation time: %f s' % (time.time() - start_time))


def compute_ds(folder):
    print('Loading images from ', folder)
    input_images, generated_images, names = \
        load_generated_images(os.path.join(folder, 'images'), 4)
    print(f'{len(input_images)} images loaded\n')

    print('Input:')
    get_detection_score(input_images)
    print('Generated')
    get_detection_score(generated_images)


if __name__ == '__main__':
    results_dir = r"D:\Networks\PoseStylizer\results\market_APS\test_latest"
    # compute_ds(results_dir)

    compute_ds(r"D:\Networks\PoseStylizer\results\market_UCCPT_100\test_180")
