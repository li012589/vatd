import torch
from torch import nn
from numpy.testing import assert_array_equal
from .flow import Flow
import numpy as np


class CombinedFlow(Flow):
    def __init__(self, flowList, prior = None, mappingBoundary=None, name = "combinedFlow"):
        super(CombinedFlow, self).__init__(prior, name)

        self.flowList = torch.nn.ModuleList(flowList)
        self.mappingBoundary = mappingBoundary

    def forward(self, z, *args, **kwargs):
        logJac = 0
        for flow in self.flowList:
            z, _logJac = flow.forward(z, *args, **kwargs)
            logJac += _logJac

        if self.mappingBoundary is not None:
            z = z * self.mappingBoundary
            logJac += np.log(self.mappingBoundary) * np.prod(z.shape[1:])
        return z, logJac

    def inverse(self, y, *args, **kwargs):
        logJac = 0
        if self.mappingBoundary is not None:
            y = y / self.mappingBoundary
            logJac -= np.log(self.mappingBoundary) * np.prod(y.shape[1:])

        for flow in self.flowList[::-1]:
            y, _logJac = flow.inverse(y, *args, **kwargs)
            logJac += _logJac

        return y, logJac
