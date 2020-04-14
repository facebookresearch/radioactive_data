# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Copy of https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py,
# with a cache to reload image paths / labels.
#

from logging import getLogger
import torch.utils.data as data

from PIL import Image

import os
import sys
import time
import pickle
from os.path import basename, dirname, join
import yaml
import torch
import numpy as np

logger = getLogger()

CACHE_DATASET = {
    '/datasets01_101/imagenet_full_size/061417/train': '/checkpoint/asablayrolles/imagenet_train.pkl',
    '/datasets01_101/imagenet_full_size/061417/val': '/checkpoint/asablayrolles/imagenet_val.pkl',
    '/datasets01_101/places205/121517/pytorch/train': '/checkpoint/asablayrolles/places205_train.pkl',
    '/datasets01_101/places205/121517/pytorch/val': '/checkpoint/asablayrolles/places205_val.pkl',
    '/private/home/asablayrolles/data/CUB_200_2011/train': '/checkpoint/asablayrolles/cub_train.pkl',
    '/private/home/asablayrolles/data/CUB_200_2011/val': '/checkpoint/asablayrolles/cub_val.pkl',
    '/private/home/rvj/data/google-landmark-2019/images/train': '/checkpoint/guismay/google-landmark-2019_images_train.pkl',
    '/private/home/rvj/data/google-landmark-2019/images/valid': '/checkpoint/guismay/google-landmark-2019_images_valid.pkl',
    '/private/home/rvj/data/google-landmark-2019/images/test': '/checkpoint/guismay/google-landmark-2019_images_test.pkl',
    '/private/home/rvj/data/google-landmark-2018/test': '/checkpoint/asablayrolles/google-landmark-2019_images_test_2018.pkl',
    "/private/home/hoss/data/google_landmarks_recognition_2019_test_final": "/checkpoint/asablayrolles/google-landmark-2019_images_test_final.pkl"
}


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):

        start = time.time()
        path_cache = CACHE_DATASET[root]
        if not os.path.isfile(path_cache):
            print("Images cache not found in %s, parsing dataset..." % path_cache,
                  file=sys.stderr)
            classes, class_to_idx = self._find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
            print("Parsing image folder took %.2f seconds." % (time.time() - start),
                  file=sys.stderr)
            with open(path_cache, "wb") as f:
                pickle.dump((classes, class_to_idx, samples), f)
        else:
            with open(path_cache, "rb") as f:
                classes, class_to_idx, samples = pickle.load(f)
            print("Loading cached images took %.2f seconds." % (time.time() - start),
                  file=sys.stderr)

        print("Dataset contains %i images." % len(samples), file=sys.stderr)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx

        self.samples = samples
        self.targets = np.array([s[1] for s in samples])

        self.transform = transform
        self.target_transform = target_transform

        # Samples are grouped by class contiguously
        assert np.all(0 <= self.targets[1:] - self.targets[:-1])
        assert np.all(self.targets[1:] - self.targets[:-1] <= 1)
        assert np.sum(self.targets[1:] - self.targets[:-1]) == max(self.targets)

        cl_positions = np.nonzero(self.targets[1:] - self.targets[:-1])[0] + 1
        cl_positions = np.insert(cl_positions, 0, 0)
        cl_positions = np.append(cl_positions, len(self.targets))

        self.class2position = {i: np.arange(cl_positions[i], cl_positions[i+1]) for i in range(len(cl_positions) - 1)}
        assert all([all(self.targets[v] == i for v in self.class2position[i]) for i in range(max(self.targets) + 1)])


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
