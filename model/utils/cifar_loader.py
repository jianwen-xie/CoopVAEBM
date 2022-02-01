from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import numpy as np
from PIL import Image
# import scipy.misc
# from skimage.measure import compare_psnr, compare_ssim
import h5py
try:
    import pickle
except ImportError:
    import cPickle as pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class CifarDataset(object):
    def __init__(self, datapath, max_dataset_size=float('inf'), shuffle=False, serial_batches=False, batch_sz=100, no_flip=False):

        self.serial_batches = serial_batches
        self.datapath = datapath

        self.images = get_cifar_dataset(self.datapath)
        # self.load_size = load_size
        # self.crop_size = crop_size
        # self.no_flip = no_flip
        print('Loaded Cifar-10 data: {}'.format(self.images.shape))

        self.num_images = min(len(self.images), max_dataset_size)


        self.num_batch = int(math.ceil(float(self.num_images) / batch_sz))
        self.batch_idx = 0
        self.batch_sz = batch_sz


        if shuffle:
            self.indices = np.random.permutation(self.num_images)
        else:
            self.indices = np.arange(self.num_images)
    
    def shuffle(self):
        np.random.shuffle(self.indices)

    # def _process_images(self, image_list):
    #     if type(image_list) == str:
    #         image_list = [image_list]
    #     images = []
    #     for img_path in image_list:
    #         images.append(preprocess_image(Image.open(img_path).convert(
    #             'RGB'), self.load_size, self.crop_size, do_flip=not self.no_flip))
    #     return np.stack(images)

    # def sample_image_pair(self):
    #     img_A = Image.open(self.A_paths[np.random.randint(0, self.A_size)]).convert('RGB')

    #     img_B = Image.open(self.B_paths[np.random.randint(0, self.B_size)]).convert('RGB')
    #     return img_A, img_B


    def get_batch(self):

        start_idx = self.batch_idx * self.batch_sz
        end_idx = min((self.batch_idx+1) * self.batch_sz, len(self))
        batch_images = self.images[start_idx: end_idx]

        self.batch_idx = self.batch_idx + 1 if end_idx < len(self) else 0

        return batch_images


    def __getitem__(self, index):
        """
        Args:
            index: np.array of batch indices
        Returns:
            img_A: batch of images in domain A
            img_B: batch of images in domain B
        """
        img = self.images[self.indices[index]]
        return img

    def __len__(self):
        return self.num_images


def preprocess_image(input_image, load_size, crop_size, do_flip=True):
    img = input_image.resize((load_size, load_size), Image.BICUBIC)
    if load_size > crop_size:
        crop_p = np.random.randint(0, load_size - crop_size, size=(2,))
        img = img.crop((crop_p[0], crop_p[1], crop_p[0] +
                        crop_size, crop_p[1] + crop_size))
    img = np.asarray(img, dtype=np.float32)
    # normalize to [-1, 1]
    img = (img / 127.5) - 1.0
    if do_flip and random.random() > 0.5:
        img = np.fliplr(img)
    return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images.sort()
    return np.array(images[:int(min(max_dataset_size, len(images)))])

def get_cifar_dataset(folder_name, name="train"):
    x = None
    y = None

    # maybe_download_and_extract()

    # folder_name = "cifar_10"

    # f = open(folder_name+'/batches.meta', 'rb')
    # f.close()

    if name is "train":
        for i in range(5):
            f = open(folder_name+'/data_batch_' + str(i + 1), 'rb')
            try:
                datadict = pickle.load(f, encoding='bytes')
            except TypeError:
                datadict = pickle.load(f)
            f.close()
            # print(datadict.keys())
            _X = datadict[b"data"]
            _Y = datadict[b'labels']

            _X = np.array(_X, dtype=float)
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X / 127.5 - 1.0
            # _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open(folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float)
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x / 127.5 - 1.0
        # x = x.reshape(-1, 32*32*3)

    return x


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def preload_images(paths, img_width=256, img_height=256, img_channel=3, mean=0.5, stddev=0.5):

    images = np.zeros((len(paths), img_width, img_height,
                       img_channel), dtype=np.float32)
    for i, img_path in enumerate(paths):
        img = Image.open(img_path).convert('RGB').resize(
            (img_width, img_height), Image.BICUBIC)
        img = np.array(img).astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img - mean) / stddev

        images[i] = img
    return images

if __name__ == '__main__':
    dataset = CifarDataset('./Input/cifar/cifar-10-batches-py/')