# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import numpy as np


class OrderedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        extra = len(self.dataset) % (num_replicas * batch_size)
        padding = num_replicas * batch_size - extra if extra > 0 else 0
        total_size = len(self.dataset) + padding
        self.indices = np.array_split(np.arange(total_size), self.num_replicas)[self.rank]
        self.indices = self.indices % len(self.dataset)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class BalancedSampler(Sampler):

    def __init__(self, data_source, balance_type="avg"):
        assert hasattr(data_source, "class2position")
        self.data_source = data_source
        self.balance_type = balance_type

        print("Using BalancedSampler with %d images per class (%s)" % (self.get_class_size(), balance_type))

    def __iter__(self):
        return iter(self.get_ids(None))

    def get_class_size(self):
        if self.balance_type == "avg":
            class_size = len(self.data_source) // len(self.data_source.class2position)
        elif self.balance_type == "max":
            class_size = max([len(v) for v in self.data_source.class2position.values()])
        elif self.balance_type == "min":
            class_size = min([len(v) for v in self.data_source.class2position.values()])
        else:
            class_size = int(self.balance_type)

        return class_size

    def get_ids(self, seed=None):
        """
        Generate a list of ids to be iterated over in the current iteration
        Typically get_ids(epoch) can be used by a DistributedSampler
        """
        sample_ids = []
        self.class_size = self.get_class_size()
        if seed is not None:
            state = np.random.RandomState(seed)
        else:
            state = np.random.RandomState()

        for cl in self.data_source.class2position.keys():
            positions = self.data_source.class2position[cl]
            replace = (len(positions) < self.class_size)
            sample_ids.extend(state.choice(positions, self.class_size, replace=replace))

        state.shuffle(sample_ids)

        return sample_ids

    def __len__(self):
        if not hasattr(self, 'class_size'):
            self.class_size = self.get_class_size()

        return self.class_size * len(self.data_source.class2position)




class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.sampler) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas


    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        sampler_idx = np.array(self.sampler.get_ids(self.epoch))
        indices = torch.randperm(len(sampler_idx), generator=g).tolist()
        print("GPU %d says first indices should be " % self.rank, indices[:10])

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        print('There are %d indices' % len(indices))

        return iter(list(sampler_idx[indices]))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch





class SeededDistributedSampler(Sampler):
    """
    Same sampler as Pytorch's but with a seed
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        print("SeededDistributedSampler, indices[:10] = ")
        print(indices[:10])

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
