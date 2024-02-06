import torch
from torch import nn
import numpy as np

from .source import Source


class DeltaSource(Source):
    def __init__(self, nvars, pos=0, trainable=False, T=1.0, name="deltaSource"):
        super().__init__(nvars, T=T, name=name)

        self.pos = torch.nn.Parameter(pos, requires_grad=trainable)

    def sample(self, batchSize, T=None):
        return self.pos.new_ones(batchSize, *self.nvars) * self.pos

    def energy(self, z):
        return self.pos.new_zeros(z.shape[0], 1)

    def logParitionFn(self, T=None):
        return self.pos.new_zeros(1, 1)