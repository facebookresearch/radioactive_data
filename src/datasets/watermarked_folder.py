# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

logger = getLogger()


class WatermarkedSet(data.Dataset):
    def __init__(self, vanilla_dataset, watermark_path, transform=None, target_transform=None):
        self.vanilla_dataset = vanilla_dataset

        watermark = torch.load(watermark_path)

        if watermark["type"] == "per_sample":
            self.marked_content = watermark["content"] # watermark["key"]

        self.transform = transform
        self.target_transform = target_transform

        num_marked = len(self.marked_content)
        per_marked = num_marked / len(vanilla_dataset)
        print(f"There are {num_marked} watermarked examples in this dataset ({100 * per_marked:.2f}%)")


    def __getitem__(self, index):
        if index in self.marked_content:
            path, target = self.marked_content[index]
            sample = np.load(path)

            assert sample.dtype == np.uint8
            assert sample.shape[2] == 3
            sample = Image.fromarray(sample)

            # assert splitext(basename(self.marked_content[index]))[0] == splitext(basename(path))[0]

            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            sample, target = self.vanilla_dataset[index]

        return sample, target


    def __len__(self):
        return len(self.vanilla_dataset)



if __name__ == "__main__":
    pass
    # folder = WatermarkedImageFolder("/datasets01_101/imagenet_full_size/061417/train")
