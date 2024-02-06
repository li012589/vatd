import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np
import utils
import flow
import source
import utils
from copy import deepcopy

from numpy.testing import assert_array_equal, assert_almost_equal


def test_concatened():
    batchSize = 100
    fixFirst = torch.tensor([[0.0]])
    L = 8
    T = torch.tensor(1.2)

    kappaMapper = utils.SoftplusWithBound(lowerBound=1e-3)
    p2 = source.VonMisesImp([1, L**2 - 1], mu = torch.randn([1, L * L - 1]), kappa = (torch.abs(torch.ones([1, L * L -1]) + 0.1 * torch.randn([1, L * L - 1]))), kappaMapper=kappaMapper)
    p1 = source.DeltaSource([1, 1], pos=fixFirst.view(-1))
    prior = source.ConcatenatedSource([p1, p2], dim=-1, reshape=[1, L, L])

    priorp = source.ConcatenatedSource([p1, p2], dim=-1, reshape=[1, L, L], directLogProb=True)

    z = prior.sample(batchSize)

    logProb = prior.logProbability(z, T=T)

    zparts1 = z.reshape(batchSize, -1)[:, 0]
    zparts2 = z.reshape(batchSize, -1)[:, 1:]

    zlogProb1 = p1.logProbability(zparts1, T=T)
    zlogProb2 = p2.logProbability(zparts2, T=T)

    assert torch.allclose(logProb, zlogProb1 + zlogProb2)

    logProbp = priorp.logProbability(z, T=T)

    assert torch.allclose(logProbp, zlogProb1 + zlogProb2)

if __name__ == "__main__":
    test_concatened()

