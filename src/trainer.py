# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
# from apex.fp16_utils import FP16_Optimizer
from .utils import repeat_to, get_optimizer


logger = getLogger()


class Trainer(object):

    def __init__(self, model, params, ftmodel=None, teacher_model=None):
        """
        Initialize trainer.
        """
        # pretrained model / model / params
        self.model = model
        self.ftmodel = ftmodel
        self.teacher_model = teacher_model
        self.params = params
        assert params.fp16 is False

        # set parameters
        self.set_parameters()

        # set optimizers
        self.set_optimizers()

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.indices = []
        self.n_iter = 0
        self.embeddings = None
        self.stats = OrderedDict(
            [('processed_i', 0)] +
            [('MSE', [])] +
            [('XE', [])] +
            [('triplet', [])] +
            [('time', [])]
        )
        self.last_time = time.time()

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}

        # all parameters
        model_params = [p for k, p in self.model.named_parameters() if p.requires_grad]
        self.parameters['model'] = model_params

        # log
        prettyNumber = lambda x: "%.2fM" % (x / 1e6) if x >= 1e6 else "%.2fK" % (x / 1e3)
        countParams = lambda x: sum([p.numel() for p in x])
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            logger.info("Number of %s parameters: %s" % (k, prettyNumber(countParams(v))))


    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}
        self.schedules = {}

        # model optimizer
        self.optimizers['model'], self.schedules['model'] = get_optimizer(self.parameters['model'], params.optimizer)

        # update schedules
        self.schedules = {
            name: None if schedule is None else repeat_to(schedule, self.params.epochs)
            for name, schedule in self.schedules.items()
        }

        # log
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def update_learning_rate(self):
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
        """
        for name in ['model', 'query', 'value']:
            schedule = self.schedules.get(name, None)
            if schedule is None:
                return
            lr = schedule[self.epoch]
            logger.info("New learning rate for %s: %f" % (name, lr))
            for param_group in self.optimizers[name].param_groups:
                param_group['lr'] = lr

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        FREQ = 5
        if self.n_iter % FREQ != 0:
            return

        s_iter = "%7i - " % self.n_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v[-FREQ:])) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])

        # learning rates
        s_lr = ""
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} images/s - ".format(self.stats['processed_i'] * 1.0 / diff)
        self.stats['processed_i'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_model(self, name):
        """
        Save the model.
        """

        path = os.path.join(self.params.dump_path, '%s.pth' % name)

        model = self.model.module if self.params.multi_gpu else self.model
        data = {
            'model': model.state_dict(),
            'epoch': self.epoch,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
            'params': vars(self.params)
        }
        for k, v in self.optimizers.items():
            data['optimizer_%s' % k] = v.state_dict()

        logger.info("Saving periodic to %s ..." % path)
        torch.save(data, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        # if not self.params.is_master:
        #     return

        data = {
            'model': self.model.state_dict(),
            'epoch': self.epoch,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
            'params': vars(self.params)
        }
        for k, v in self.optimizers.items():
            data['optimizer_%s' % k] = v.state_dict()

        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint-%d.pth' % self.params.global_rank)
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            return
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.params.local_rank))

        # reload model parameters
        if self.params.multi_gpu:
            self.model.load_state_dict(data['model'])
        else:
            state_dict = {k.replace("module.", ""): v for k, v in data['model'].items()}
            self.model.load_state_dict(state_dict)

        # reload optimizers
        for k, v in self.optimizers.items():
            self.optimizers[k].load_state_dict(data['optimizer_%s' % k])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_model('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return

        for metric, biggest in self.metrics:
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None:
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                exit()

        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        self.save_checkpoint()
        self.epoch += 1

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # backward / optimization
        for name in self.optimizers.keys():
            self.optimizers[name].zero_grad()

        if self.params.fp16:
            list(self.optimizers.values())[0].backward(loss)
        else:
            loss.backward()

        for name in self.optimizers.keys():
            self.optimizers[name].step()


    def classif_step(self, images, targets):
        """
        Classification step.
        """
        start = time.time()
        params = self.params
        self.model.train()

        # batch
        images = images.cuda(non_blocking=True).half() if params.fp16 else images.cuda(non_blocking=True)
        if self.ftmodel is not None:
            with torch.no_grad():
                images = self.ftmodel(images)

        # forward / loss / optimize
        output = self.model(images)

        loss = F.cross_entropy(output, targets.cuda(non_blocking=True), reduction='mean')
        self.optimize(loss)

        # statistics
        self.stats['processed_i'] += params.batch_size
        self.stats['XE'].append(loss.item())
        self.stats['time'].append(time.time() - start)


    def distillation_step(self, images, targets):
        """
        Distillation step.
        """
        start = time.time()
        params = self.params
        self.model.train()

        # batch
        images = images.cuda(non_blocking=True).half() if params.fp16 else images.cuda(non_blocking=True)

        teacher_prelogits = self.teacher_model(images)
        targets = F.softmax(teacher_prelogits / params.temperature, dim=1)

        student_prelogits = self.model(images)
        student_logits = F.log_softmax(student_prelogits / params.temperature, dim=1)

        loss = - (student_logits * targets).sum(dim=1).mean()
        self.optimize(loss)

        # statistics
        self.stats['processed_i'] += params.batch_size
        self.stats['XE'].append(loss.item())
        self.stats['time'].append(time.time() - start)


    def get_scores(self):
        scores = {
            "speed": self.params.batch_size / np.mean(self.stats['time']),
        }

        for k in self.stats.keys():
            if type(self.stats[k]) is list and len(self.stats[k]) >= 1:
                scores[k] = np.mean(self.stats[k])

        return scores
