# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.utils.data import Dataset


class DataAugmenter(Dataset):

    def __init__(self, dataset, augmentation, postprocess, seed=0):
        self.dataset = dataset
        self.augmentation = augmentation
        self.seed = seed
        self.postprocess = postprocess

    def __getitem__(self, index):
        img, label = self.dataset[index]
        aug_params = self.augmentation.sample_params(img, seed=self.seed + index)
        img_augmented = self.augmentation(img, aug_params)
        img_augmented = self.postprocess(img_augmented)

        return img_augmented, label

    def __len__(self):
        return len(self.dataset)
