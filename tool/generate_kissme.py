import math
import sys
import random
import glob
import os

import scipy.io
from skimage import color, io
from skimage.feature import SIFT
from skimage.exposure import histogram
import numpy as np

X = np.ones((18, 6))


def kernelmatrix(X, X2, sigma):
    n1sq = (X ** 2).sum(axis=0, keepdims=True)
    n1 = X.shape[1]

    if not X2:
        K = (np.ones((n1, 1)) @ n1sq).transpose() + np.ones((n1, 1)) @ n1sq - 2 * (X.transpose() @ X)
    else:
        n2sq = (X2 ** 2).sum(axis=0, keepdims=True)
        n2 = X2.shape[1]
        K = (np.ones((n2, 1)) @ n1sq).transpose() + np.ones((n1, 1)) @ n2sq - 2 * X.transpose() @ X2

    K = np.exp(-K * sigma)

    return K


def computeT(n, e, H, K):
    I = (1 / (n * e ** 2)) * H @ np.linalg.inv(np.eye(len(K)) + (1 / (n * e)) * K @ H)
    return I


def computeH(n, ib, ie):
    B = np.asarray([np.count_nonzero(ib == t) for t in range(n)])
    B = np.diag(B)
    E = np.asarray([np.count_nonzero(ie == t) for t in range(n)])
    E = np.diag(E)

    W = np.zeros((n, n))

    for i in range(len(ib)):
        W[ib[i], ie[i]] = W[ib[i], ie[i]] + 1

    H = B + E - W.transpose() - W
    return H


def projectPSD(M):
    val, vec = np.linalg.eig(M)
    D = np.linalg.inv(vec) @ M @ vec
    V = vec.real
    d = np.diagonal(D.real).copy()
    d[d <= 0] = sys.float_info.epsilon
    M = V @ np.diag(d) @ V.transpose()
    return M


def kernel(e, S, D, K, pmetric):
    n = K.shape[0]
    n1 = S.shape[1]
    n0 = D.shape[1]
    H0 = computeH(n, D[0, :], D[1, :])
    H1 = computeH(n, S[0, :], S[1, :])
    C = computeT(n0, e, H0, K) - computeT(n1, e, H1, K)
    if pmetric:
        C = projectPSD(C)
    return C


def get_histo(img):
    shape = img.shape
    if len(shape) > 2:
        histo = np.concatenate([get_histo(img[..., i]) for i in range(shape[-1])])
        return histo
    else:
        histo = histogram(img, nbins=30)[0].astype(float)
        histo /= img.size
        return histo


def get_features(img, thresh=5):
    lab = color.rgb2lab(img)
    histo_lab = get_histo(lab)

    hsv = color.rgb2hsv(img)
    histo_hsv = get_histo(hsv)

    feats = [histo_lab, histo_hsv]

    rng = np.random.default_rng()
    sift_extractor = SIFT()
    for i in range(3):
        try:
            sift_extractor.detect_and_extract(lab[:, :, i])

            sift_descr = sift_extractor.descriptors
            sift_descr = sift_descr[rng.choice(len(sift_descr), size=thresh, replace=len(sift_descr) < thresh)]
        except RuntimeError:
            sift_descr = np.zeros(thresh * 128)
        feats.append(sift_descr.reshape((-1)))
    return np.concatenate(feats)


def get_patches(img):
    nb_patch = 6
    h_patch = img.shape[0] // nb_patch
    thresh = [0] + [h_patch * (i + 1) for i in range(nb_patch - 1)] + [img.shape[0]]
    patches = [img[t:thresh[i + 1]] for i, t in enumerate(thresh[:-1])]
    return patches


# d =3, ft = 2
# X = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [5, 6, 5, 6]])
# S = np.array([[0, 1, 2], [0, 1, 3]])
# D = np.array([[0, 1, 3], [2, 3, 1]])
if False:
    fold = "D:/Networks/PoseStylizer/dataset/fashion_data/train/"
    ldir = glob.glob(f"{fold}*.jpg")
    actors = [os.path.split(l)[1].split('_')[0] for l in ldir]
    actors = list(dict([(a, None) for a in actors]))
    actors_100 = random.sample(actors, 80)
    xs = []
    for act in actors_100:
        l_act = glob.glob(f"{fold}{act}_*.jpg")
        if len(l_act) < 3:
            continue
        for i, img_path in enumerate(random.sample(l_act, 3)):
            img = io.imread(img_path)
            patches = get_patches(img)
            feats = [get_features(p) for p in patches]
            xs.append(feats)

    x = np.array(xs).swapaxes(0, 1).swapaxes(1, 2)
    np.save('temp_100.npy', x)
else:
    x = np.load('temp_100.npy').swapaxes(1, 2)

idx = list(range(x.shape[1]))

random.shuffle(idx)
idx_s = [((i + (1 if random.random() > 0.5 else -1)) - i // 3 * 3) % 3 + i // 3 * 3 for i in idx]
s = np.array([idx, idx_s])

idx_d = idx.copy()
random.shuffle(idx_d)
d = np.array([idx, idx_d])

mat = scipy.io.loadmat(r"C:\Users\Romain Guesdon\Downloads\test.mat")
X = mat['X']
S = mat['S'] - 1
D = mat['D'] - 1
m = mat['M']
k = mat['K']

# K = kernelmatrix(X, [], 2 ** -16)
# print(np.sum(K - k))
#
# M = kernel(0.001, S, D, K, 1)
# print(np.sum(M - m))
ks = []
ms = []
for i, xi in enumerate(x):
    k = kernelmatrix(xi, [], 2 ** -16)
    m = kernel(0.001, s, d, k, 1)
    ks.append(k)
    ms.append(m)

print()