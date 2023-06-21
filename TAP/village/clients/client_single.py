"""Single model default victim class."""

import torch
import numpy as np
from collections import defaultdict


from ..utils import set_random_seed
from ..consts import BENCHMARK
import pdb

torch.backends.cudnn.benchmark = BENCHMARK

from .client_base import _ClientBase


class _ClientSingle(_ClientBase):
    def initialize(self, seed=None, feature_extractor=False):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        (
            self.model,
            self.defs,
            self.criterion,
            self.optimizer,
            self.scheduler,
        ) = self._initialize_model(
            self.args.net[0], feature_extractor=feature_extractor
        )

        self.model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _iterate(self, furnace, poison_delta, max_epoch=None):
        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs

        def loss_fn(model, outputs, labels):
            return self.criterion(outputs, labels)

        single_setup = (
            self.model,
            self.defs,
            self.criterion,
            self.optimizer,
            self.scheduler,
        )
        for self.epoch in range(max_epoch):
            self._step(furnace, poison_delta, loss_fn, self.epoch, stats, *single_setup)
            if self.args.dryrun:
                break
        return stats

    def step(self, furnace, poison_delta, poison_targets, true_classes):
        stats = defaultdict(list)

        def loss_fn(model, outputs, labels):
            normal_loss = self.criterion(outputs, labels)
            model.eval()
            if self.args.adversarial != 0:
                target_loss = (
                    1
                    / self.defs.batch_size
                    * self.criterion(model(poison_targets), true_classes)
                )
            else:
                target_loss = 0
            model.train()
            return normal_loss + self.args.adversarial * target_loss

        single_setup = (self.model, self.criterion, self.optimizer, self.scheduler)
        self._step(furnace, poison_delta, loss_fn, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            (
                self.model,
                self.criterion,
                self.optimizer,
                self.scheduler,
            ) = self._initialize_model()
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        def apply_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()

        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        _, _, self.optimizer, self.scheduler = self._initialize_model()

    def gradient(self, images, labels, criterion=None, batched=False):
        if criterion is None:
            loss = self.criterion(self.model(images), labels)
        else:
            loss = criterion(self.model(images), labels)
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)

    def compute(self, function, *args):
        return function(self.model, self.criterion, self.optimizer, *args)
