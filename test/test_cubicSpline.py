from flowRelated import*

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import utils
import flow
import source
import pytest


def test_bijective():
    K = 5
    B = 3
    p = source.Uniform([1, 8, 8], -B, B)
    T = torch.tensor(1.2)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(1 * 8 * 4), torch.ones(1 * 8 * 4)])[torch.randperm(1 * 8 * 8)].reshape(1, 1, 8, 8)
        else:
            b=1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    networkList = [utils.SimpleMLPreshape([1*8*4, 50, 50, (2*K+2)*1*8*4], [nn.ELU(), nn.ELU(), None], reshapeBack=True, shape=[-1,2 * K + 2, 8, 4]) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, prior=p).double()

    samples, logprob = f.sample(100)

    outBound = ((samples < -B) * (samples > B)).sum()
    assert outBound == 0
    assert torch.isnan(logprob).sum() == 0
    assert torch.isinf(logprob).sum() == 0

    bijective(f, decimal=4)

    # test CNN
    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(1 * 8 * 4), torch.ones(1 * 8 * 4)])[torch.randperm(1 * 8 * 8)].reshape(1, 1, 8, 8)
        else:
            b=1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    networkList = [utils.ConvResidualNet(1, 2 * K + 2, 10, 4, activation=F.relu) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, prior=p).double()

    samples, logprob = f.sample(100)

    outBound = ((samples < -B) * (samples > B)).sum()

    assert outBound == 0
    assert torch.isnan(logprob).sum() == 0
    assert torch.isinf(logprob).sum() == 0

    bijective(f, decimal=4)

    # test temperature as a factor
    networkList = [utils.ConvResidualNet(2, 2 * K + 2, 10, 4, activation=F.relu) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, prior=p).double()

    samples, logprob = f.sample(100, T=T)

    outBound = ((samples < -B) * (samples > B)).sum()

    assert outBound == 0
    assert torch.isnan(logprob).sum() == 0
    assert torch.isinf(logprob).sum() == 0

    bijective(f, decimal=4, args=[T])

    # test fixFirst option
    L = 8
    fixFirst = torch.zeros(1, 1).double()
    p1 = source.DeltaSource([1, 1], pos=fixFirst.view(-1))
    p2 = source.UniformImp([1, L**2 - 1], low=-B, high=B)
    prior = source.ConcatenatedSource([p1, p2], dim=-1, reshape=[1, L, L])

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.ones(1 * 8 * 4), torch.zeros(1 * 8 * 4)]).reshape(1, 1, 8, 8)
        else:
            b = 1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    networkList = [utils.ConvResidualNet(2, 2 * K + 2, 10, 4, activation=F.relu) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, fixFirst=fixFirst, prior=prior).double()

    samples, logprob = f.sample(100, T=T)

    assert torch.allclose(samples[:, :, 0, 0], fixFirst)

    outBound = ((samples < -B) * (samples > B)).sum()

    assert outBound == 0
    assert torch.isnan(logprob).sum() == 0
    assert torch.isinf(logprob).sum() == 0

    bijective(f, decimal=4, args=[T])


def test_detJacbian():
    K = 5
    B = 3
    T = torch.tensor(1.2)
    p = source.Uniform([1, 8, 8], -B, B)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(1 * 8 * 4), torch.ones(1 * 8 * 4)])[torch.randperm(1 * 8 * 8)].reshape(1, 1, 8, 8)
        else:
            b=1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    networkList = [utils.SimpleMLPreshape([1*8*4, 50, 50, (2*K + 2)*1*8*4], [nn.ELU(), nn.ELU(), None], reshapeBack=True, shape=[-1, 2 * K + 2, 8, 4]) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, prior=p).double()

    detJacobian(f, decimal=4)

    # test CNN
    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(1 * 8 * 4), torch.ones(1 * 8 * 4)])[torch.randperm(1 * 8 * 8)].reshape(1, 1, 8, 8)
        else:
            b=1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    networkList = [utils.ConvResidualNet(1, 2 * K + 2, 10, 4, activation=F.relu) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, prior=p).double()

    detJacobian(f, decimal=3)

    # test temperature as a factor
    networkList = [utils.ConvResidualNet(2, 2 * K + 2, 10, 4, activation=F.relu) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, prior=p).double()
    detJacobian(f, decimal=3, args=[T])

    # test fixFirst option
    L = 8
    fixFirst = torch.zeros(1, 1).double()
    p1 = source.DeltaSource([1, 1], pos=fixFirst.view(-1))
    p2 = source.UniformImp([1, L**2 - 1], low=-B, high=B)
    prior = source.ConcatenatedSource([p1, p2], dim=-1, reshape=[1, L, L])

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.ones(1 * 8 * 4), torch.zeros(1 * 8 * 4)]).reshape(1, 1, 8, 8)
        else:
            b = 1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    networkList = [utils.ConvResidualNet(2, 2 * K + 2, 10, 4, activation=F.relu) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, fixFirst=fixFirst, prior=prior).double()

    detJacobian(f, decimal=3, args=[T])


def test_saveload():
    K = 5
    B = 3
    p = source.Uniform([1, 8, 8], -B, B)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(1 * 8 * 4), torch.ones(1 * 8 * 4)])[torch.randperm(1 * 8 * 8)].reshape(1, 1, 8, 8)
        else:
            b=1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    networkList = [utils.SimpleMLPreshape([1*8*4, 50, 50, (2*K+2)*1*8*4], [nn.ELU(), nn.ELU(), None], reshapeBack=True, shape=[-1, 2 * K + 2, 8, 4]) for _ in range(4)]

    f = flow.CubicSpline(networkList, maskList, K=K, B=B, prior=p).double()

    K = 5
    B = 3
    p = source.Uniform([1, 8, 8], -B, B)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(1 * 8 * 4), torch.ones(1 * 8 * 4)])[torch.randperm(1 * 8 * 8)].reshape(1, 1, 8, 8)
        else:
            b=1 - b
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.float32)
    networkList = [utils.SimpleMLPreshape([1*8*4, 50, 50, (2*K+2)*1*8*4], [nn.ELU(), nn.ELU(), None], reshapeBack=True, shape=[-1, 2 * K + 2, 8, 4]) for _ in range(4)]

    blankf = flow.CubicSpline(networkList, maskList, K=K, B=B, prior=p).double()

    saveload(f, blankf, decimal=3)


if __name__ == "__main__":
    test_bijective()
    test_detJacbian()
    test_saveload()