# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger
import torch
from torchvision import transforms
from torch.utils.data import Subset
import numpy as np

from .datasets import TargetTransformDataset
from .datasets.folder import ImageFolder
from .datasets.sampler import DistributedSampler, SeededDistributedSampler
from .datasets.watermarked_folder import WatermarkedSet
from .data_augmentations import CenterCrop, RandomResizedCropFlip, DifferentiableDataAugmentation

logger = getLogger()

DATASETS = {
    'cifar10': {
        'train': '/private/home/asablayrolles/data/radioactive/cifar10/',
        'valid': '/private/home/asablayrolles/data/radioactive/cifar10/vanilla_test',
        # 'test': '/private/home/asablayrolles/data/radioactive/cifar10/vanilla_test',
        'num_classes': 10,
        'img_size': 40,
        'crop_size': 32,
    },
    'imagenet': {
        'train': '/datasets01_101/imagenet_full_size/061417/train',
        'valid': '/datasets01_101/imagenet_full_size/061417/val',
        'num_classes': 1000,
        'img_size': 256,
        'crop_size': 224,
    },
    'places205': {
        'train': '/datasets01_101/places205/121517/pytorch/train',
        'valid': '/datasets01_101/places205/121517/pytorch/val',
        'num_classes': 205,
        'img_size': 256,
        'crop_size': 224,
    }
}

SUBCLASSES = {
    "imagenet": {
        n_cl: list(np.load("/private/home/asablayrolles/data/radioactive/imagenet_classes/%d.npy" % n_cl)) for n_cl in [10, 20, 50, 100, 200, 500]
    }
}

def populate_dataset(params):
    assert params.dataset in DATASETS

    if params.num_classes == -1:
        params.num_classes = DATASETS[params.dataset]['num_classes']
    params.img_size = DATASETS[params.dataset]['img_size']
    params.crop_size = DATASETS[params.dataset]['crop_size']


NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
NORMALIZE_CIFAR = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


def getCifarTransform(name, img_size=40, crop_size=32, as_list=False, normalization=True):
    assert name in ["center", "flip", "random"]

    padding = (img_size - crop_size) // 2
    if name == "random":
        transform = [
            # lambda x: random_crop(x, crop_size=crop_size, padding=padding),
            # random_hflip,
            transforms.ToTensor()
        ]
    elif name == "flip":
        transform = [
            # random_hflip,
            transforms.ToTensor()
        ]
    else:
        transform = [
            transforms.ToTensor()
        ]

    if normalization:
        transform.append(NORMALIZE_CIFAR)

    if as_list:
        return transform
    else:
        return transforms.Compose(transform)


def getImagenetTransform(name, img_size=256, crop_size=224, normalization=True, as_list=False, differentiable=False):
    transform = []
    if differentiable:
        if name == "random":
            transform = RandomResizedCropFlip(crop_size)
        elif name == "center":
            transform = CenterCrop(img_size, crop_size)
        else:
            assert name == "none"
            transform = DifferentiableDataAugmentation()
    else:
        if name == "random":
            transform = [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
            ]
        elif name == "tencrop":
            transform = [
                transforms.Resize(img_size),
                transforms.TenCrop(crop_size),
            ]
        elif name == "center":
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
            ]
        else:
            assert name == "none"

    if name == "tencrop":
        postprocess = [
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops])
        ]
    else:
        postprocess = [
            transforms.ToTensor()
        ]

    if normalization:
        if name == "tencrop":
            postprocess.append(transforms.Lambda(lambda crops: torch.stack([NORMALIZE_IMAGENET(crop) for crop in crops])))
        else:
            postprocess.append(NORMALIZE_IMAGENET)

    if as_list:
        return transform + postprocess
    else:
        if differentiable:
            return transform, transforms.Compose(postprocess)
        else:
            return transforms.Compose(transform + postprocess)



def get_data_loader(params, split, transform, shuffle, distributed_sampler, watermark_path=""):
    """
    Get data loader over imagenet dataset.
    """
    assert params.dataset in DATASETS
    # assert (start is None) == (size is None) # either both are None, or neither
    if params.num_classes == -1:
        params.num_classes = DATASETS[params.dataset]['num_classes']
    class_mapper = lambda x: x

    # Transform
    if params.dataset.startswith("cifar") or params.dataset == "tiny" or params.dataset == "mini_imagenet":
        transform = getCifarTransform(transform, img_size=params.img_size, crop_size=params.crop_size, normalization=True)
    elif params.dataset in ["imagenet", "flickr", "cub", "places205"]:
        transform = getImagenetTransform(transform, img_size=params.img_size, crop_size=params.crop_size, normalization=True)

    # Data
    if params.dataset in ["cifar10", "mini_imagenet"]:
        pass
        # if split == "valid":
        #     if data_path == "":
        #         data = CIFAR10(root=DATASETS[params.dataset][split], transform=transform, return_index=return_index, overlay=overlay, blend_type=blend_type, alpha=alpha, overlay_class=overlay_class)
        #     else:
        #         data = CIFAR10(root=join(dirname(DATASETS[params.dataset][split]), data_path), transform=transform, return_index=return_index, overlay=overlay, blend_type=blend_type, alpha=alpha, overlay_class=overlay_class)
        # else:
        #     data = CIFAR10(root=join(DATASETS[params.dataset][split], data_path), transform=transform, return_index=return_index, overlay=overlay, blend_type=blend_type, alpha=alpha, overlay_class=overlay_class)
    elif params.dataset in ["imagenet", "places205"]:
        vanilla_data = ImageFolder(root=DATASETS[params.dataset][split], transform=transform)
        if watermark_path != "":
            data = WatermarkedSet(vanilla_data, watermark_path=watermark_path, transform=transform)
        else:
            data = vanilla_data
    else:
        raise NotImplementedError()


    # Restricted the number of classes, remap them to [0, n_cl - 1]
    if params.num_classes != DATASETS[params.dataset]['num_classes']:
        indices = []
        for cl_id in SUBCLASSES[params.dataset][params.num_classes]:
            indices.extend(data.class2position[cl_id])

        data = Subset(data, indices)
        if params.num_classes != DATASETS[params.dataset]['num_classes']:
            class_mapper = lambda i: SUBCLASSES[params.dataset][params.num_classes].index(i)
            data = TargetTransformDataset(data, class_mapper)

    # sampler
    sampler = None
    if distributed_sampler:
        if sampler is None:
            # sampler = torch.utils.data.distributed.DistributedSampler(data)
            sampler = SeededDistributedSampler(data, seed=params.seed)
        else:
            sampler = DistributedSampler(sampler)


    # data loader
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=params.batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=params.nb_workers,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader, sampler, class_mapper
