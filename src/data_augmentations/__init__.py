# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import random
import torch.nn.functional as F
import torch
from PIL import Image
import torchvision.transforms.functional as F_img

PIL_INTERPOLATION = {
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "box": Image.BOX,
    "hamming": Image.HAMMING
}

class DifferentiableDataAugmentation:
    def __init__(self):
        pass

    def sample_params(self, x, seed=None):
        # Sample parameters for a given data augmentation
        return 0

    def apply(self, x, params):
        # Apply data augmentation to image
        assert params == 0

        return x

    def __call__(self, x, params):
        return self.apply(x, params)


class CenterCrop(DifferentiableDataAugmentation):

    def __init__(self, resize, crop_size, interpolation='bilinear'):
        assert resize > crop_size
        self.resize = resize
        self.crop_size = crop_size
        self.half_size = crop_size // 2
        self.interpolation = interpolation

    def sample_params(self, x, seed=None):
        return 0

    def apply(self, x, augmentation):
        assert augmentation == 0

        if type(x) is torch.Tensor:
            assert len(x.size()) == 4
            min_dim = min(x.size()[2:])
            scale = self.resize / min_dim

            x_resized = F.interpolate(x, scale_factor=scale, mode=self.interpolation)
            x_resized = x_resized.clamp(min=x.min().item(), max=x.max().item())

            i_center, j_center = x_resized.size(2) // 2, x_resized.size(3) // 2

            return x_resized[..., i_center - self.half_size:i_center + self.half_size, j_center - self.half_size:j_center + self.half_size]
        else:
            x = F_img.resize(x, self.resize, PIL_INTERPOLATION[self.interpolation])

            return F_img.center_crop(x, self.crop_size)


class RandomResizedCropFlip(DifferentiableDataAugmentation):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), flip=True, interpolation='bilinear'):
        assert len(ratio) == 2
        assert len(scale) == 2

        self.ratio = ratio
        self.scale = scale
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        self.flip = flip
        self.interpolation = interpolation


    def sample_params(self, x, seed: int=None):
        if type(x) is torch.Tensor:
            assert len(x.size()) == 4
            width, height = x.size(3), x.size(2)
        elif type(x) is Image.Image:
            width, height = x.size

        if seed is not None:
            random.seed(seed)

        flip = random.randint(0, 1) if self.flip else 0
        area = width * height

        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, flip

        # Fallback to central crop
        in_ratio = width / height
        if (in_ratio < min(self.ratio)):
            w = width
            h = int(round(w / min(self.ratio)))
        elif (in_ratio > max(self.ratio)):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2

        return i, j, h, w, flip


    def apply(self, x, augmentation):
        i, j, h, w, flip = augmentation

        if type(x) is torch.Tensor:
            assert len(x.size()) == 4
            x_resized = F.interpolate(x[..., i:(i+h), j:(j+w)], size=self.size, mode=self.interpolation)
            x_resized = x_resized.clamp(min=x.min().item(), max=x.max().item())

            if flip:
                x_resized = x_resized[..., torch.arange(x_resized.size(-1) - 1, -1, -1)]
        else:
            x_resized = F_img.resized_crop(x, i, j, h, w, self.size, PIL_INTERPOLATION[self.interpolation])
            if flip:
                x_resized = F_img.hflip(x_resized)

        return x_resized


if __name__ == "__main__":
    x = torch.zeros(1, 3, 375, 500)

    da = CenterCrop(256, 224)
    da_params = da.sample_params(x)

    da.apply(x, da_params)
