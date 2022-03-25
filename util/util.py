from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import subprocess as sp
import gc

from skimage.draw import disk, line_aa, polygon, polygon2mask


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, index=0):
    image_numpy = image_tensor[index].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


# draw pose img
LIMB_SEQ = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
            [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
            [0, 15], [15, 17], [2, 16], [5, 17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
          'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1


def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map[..., :18]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis=(0, 1)),
                                      pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)


def draw_pose_from_map(pose_map, threshold=0.1, index=0, **kwargs):
    # CHW -> HCW -> HWC
    pose_map = pose_map[index].cpu().transpose(1, 0).transpose(2, 1).numpy()

    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_pose_from_cords(cords, pose_map.shape[:2], **kwargs)


def draw_pose_from_map_wider(pose_map, threshold=0.1, ratio=0.225, **kwargs):
    # CHW -> HCW -> HWC
    pose_map = pose_map[0].cpu().transpose(1, 0).transpose(2, 1).numpy()

    cords = map_to_cord(pose_map, threshold=threshold)
    for i in range(len(cords)):
        tmp = cords[i][0] * ratio
        if tmp < 0:
            tmp = -1
        cords[i][0] = round(tmp)
        tmp = cords[i][1] * ratio
        if tmp < 0:
            tmp = -1
        cords[i][1] = round(tmp)
    return draw_pose_from_cords(cords, (round(pose_map.shape[0] * ratio), round(pose_map.shape[1] * ratio)), **kwargs)


# draw pose from map
def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3,), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = disk((joint[0], joint[1]), radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def avg_dic(old_dic, new_dic, n_iter):
    if n_iter < 1:
        return new_dic
    for k, v in enumerate(old_dic):
        if k in new_dic:
            new_dic[k] = (v * (n_iter - 1) + new_dic[k]) / n_iter

    return new_dic


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def debug_gpu_memory(model):
    print("Free memory: ", get_gpu_memory())
    attribs = {k: v.nelement() * v.element_size() for k, v in model.__dict__.items() if
               isinstance(v, torch.Tensor)}

    with open('garbage.log', 'a') as f:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    f.write(f'{type(obj)}: {obj.size()}\n')
            except:
                pass

        f.write('%' * 20)
        for n, t in model.named_parameters():
            f.write(f'{n}: {t.shape}\n')

        f.write('%' * 20)
        for m in [model, model.netG]:
            for n, t in m.__dict__.items():
                if torch.is_tensor(t):
                    f.write(f'{n}: {t.shape}\n')


def mask_from_pose(pose):
    thresh = 0.5
    masks = pose.sum(dim=1, keepdims=True) > 0.7
    rad = (pose > thresh).sum(dim=-1).max().item()

    img_size = pose.shape[-2:]
    v, kps = pose.view(*pose.shape[:-2], -1).max(dim=-1)
    kps = torch.stack((kps.div(img_size[1], rounding_mode='trunc'), kps % img_size[1]), -1)
    v = v > 0.5

    for b, kp in enumerate(kps):
        points = [kp[i].tolist() for i in [0, 14, 16, 2, 3, 4, 8, 9, 10, 13, 12, 11, 7, 6, 5, 17, 15] if v[b, i]]
        poly = polygon2mask(list(img_size), points)
        for p in range(len(points)):
            pt1 = points[p]
            pt2 = points[(p + 1) % len(points)]
            if pt1 == pt2:
                continue
            y, x, _ = weighted_line(*pt1, *pt2, rad)
            in_bound = (x >= 0) * (x < img_size[1]) * (y >= 0) * (y < img_size[0])
            x = x[in_bound]
            y = y[in_bound]
            poly[y, x] = True
        masks[b] += torch.tensor(poly, device=pose.device)

    return masks


def trapezium(y, y0, w):
    return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def weighted_line(r0, c0, r1, c1, w, r_min=0, r_max=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1 - c0) < abs(r1 - r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, r_min=r_min, r_max=r_max)
        return yy, xx, val

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, r_min=r_min, r_max=r_max)

    # The following is now always < 1 in abs
    slope = (r1 - r0) / (c1 - c0)

    # Adjust weight by the slope
    w *= np.sqrt(1 + np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1 + 1, dtype=float)
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w / 2)
    yy = (np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(1, -1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapezium(yy, y.reshape(-1, 1), w).flatten()
    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside the picture
    mask = np.logical_and.reduce((yy >= r_min, yy < r_max, vals > 0))

    return yy[mask].astype(int), xx[mask].astype(int), vals[mask]
