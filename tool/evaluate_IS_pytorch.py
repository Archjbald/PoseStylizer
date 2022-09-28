import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import os

import skimage
from PIL import Image

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def load_generated_images(images_folder, len_img, idx_fake):
    input_images = []
    generated_images = []

    names = []
    img_list = os.listdir(images_folder)
    for img_name in img_list:
        img = skimage.io.imread(os.path.join(images_folder, img_name))
        w = int(img.shape[1] / len_img)  # h, w ,c

        imgs = [img[:, i * w: (i + 1) * w] for i in range(len_img)]

        input_images += [imgs[0]]
        generated_images += [imgs[idx_fake]]

        assert img_name.endswith('_vis.png') or img_name.endswith(
            '_vis.jpg'), 'unexpected img name: should end with _vis.png'
        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        fr = img_name[0]
        to = img_name[1]

        names.append([fr, to])

    Image.fromarray(generated_images[0]).save(os.path.join(images_folder, '../sample_output.png'))

    return (np.stack(input_images, axis=0), np.stack(generated_images, axis=0), names)


if __name__ == '__main__':
    import torchvision.datasets as dset
    import torchvision.transforms as transforms


    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        def __getitem__(self, index):
            x = self.data[index]
            if self.transform:
                x = self.transform(x)
            return x

        def __len__(self):
            return len(self.data)


    cifar = dset.CIFAR10(root='data/', download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor(),
                             transforms.GaussianBlur(9),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
                         )

    cifar = IgnoreLabelDataset(cifar)

    # results_dir = r"D:\Networks\PoseStylizer\results\market_APS\test_latest"
    # print('Loading images from ', results_dir)
    # input_images, generated_images, names = \
    #     load_generated_images(os.path.join(results_dir, 'images'), 5, 4)
    # print(f'{len(input_images)} images loaded\n')
    #
    # market = MyDataset(input_images)

    print("Calculating Inception Score...")
    print(inception_score(cifar, cuda=True, batch_size=32, resize=True, splits=10))
    # print(inception_score(MyDataset(generated_images), cuda=True, batch_size=32, resize=True, splits=10))
