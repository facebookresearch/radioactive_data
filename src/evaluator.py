# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch


logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.model = trainer.model
        self.ftmodel = trainer.ftmodel
        self.params = params

    def run_all_evals(self, trainer, evals, data_loader, *args, **kwargs):
        """
        Run all evaluations.
        """
        assert type(evals) is list
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():

            if evals is None or 'classif' in evals:
                for data_type in ['valid']:
                    self.eval_classif(data_type, scores, data_loader)

        return scores


    def eval_classif(self, data_type, scores, data_loader):
        """
        Evaluate classification.
        """
        params = self.params
        self.model.eval()

        # stats
        accuracies = []

        # memories
        topk = [1, 5, 10, 20, 50, 100, 200, 500]
        topk = [k for k in topk if k <= params.num_classes]

        for images, targets in data_loader:
            images = images.cuda().half() if params.fp16 else images.cuda()
            if self.ftmodel is not None:
                images = self.ftmodel(images)

            output = self.model(images)
            accuracies.append(accuracy(output.cpu(), targets, topk=tuple(topk)))

        # accuracy
        for i_k, k in enumerate(topk):
            scores['top%d_acc' % k] = np.mean([x[i_k] for x in accuracies])



def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
