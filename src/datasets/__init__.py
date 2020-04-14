# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.utils.data import Dataset


class TargetTransformDataset(Dataset):
    def __init__(self, dataset, target_transform):
        self.dataset = dataset
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample, target = self.dataset[index]

        return sample, self.target_transform(target)

    def __len__(self):
        return len(self.dataset)
