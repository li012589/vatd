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
from utils import ive
from copy import deepcopy

from numpy.testing import assert_array_equal, assert_almost_equal


def vmMean(mu, kappa):
    return torch.cos(mu) * (ive(1, kappa) / ive(0, kappa))

def test_sampleImp():
    batchSize = 100000
    nvars = [2, 8, 8]
    B = 1.0
    T = torch.tensor(4.0)

    mu = (torch.arange(np.prod(nvars)).reshape(nvars).to(torch.float32) / np.prod(nvars) - 0.5) * 2 * 2.7
    kappa = (torch.arange(np.prod(nvars)) * 0.01).reshape(nvars).to(torch.float32) + 1
    kappa = kappa * 10

    p = source.VonMisesImp(nvars, mu, kappa, T=1.0)

    meanT = vmMean(mu, kappa)

    samples = p.sample(batchSize)
    mean = torch.cos(samples).mean(0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT.detach().numpy(), decimal=2)

    # test temperature
    p = source.VonMisesImp(nvars, mu, kappa, T=T)

    meanT = vmMean(mu, kappa/T)

    samples = p.sample(batchSize)
    mean = torch.cos(samples).mean(0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT.detach().numpy(), decimal=2)

    # test boundary
    p = source.VonMisesImp(nvars, mu, kappa, B=B, T=T)

    meanT = vmMean(mu, kappa/T)

    samples = p.sample(batchSize)
    mean = torch.cos(samples / B * np.pi).mean(0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT.detach().numpy(), decimal=2)


    #test T with batchSize
    p = source.VonMisesImp(nvars, mu, kappa, T=T.repeat(batchSize))

    meanT = vmMean(mu, kappa/T)

    samples = p.sample(batchSize)
    mean = torch.cos(samples).mean(0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT.detach().numpy(), decimal=2)


def test_sample():
    batchSize = 100000
    nvars = [2, 8, 8]
    T = torch.tensor(4.0)

    mu = (torch.arange(np.prod(nvars)).reshape(nvars).to(torch.float32) / np.prod(nvars) - 0.5) * 2 * 2.7
    kappa = (torch.arange(np.prod(nvars)) * 0.01).reshape(nvars).to(torch.float32) + 1
    kappa = kappa * 10

    p = source.VonMises(nvars, mu, kappa, T=1.0)

    meanT = vmMean(mu, kappa)

    samples = p.sample(batchSize)
    mean = torch.cos(samples).mean(0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT.detach().numpy(), decimal=2)

    # test temperature
    p = source.VonMises(nvars, mu, kappa, T=T)

    meanT = vmMean(mu, kappa/T)

    samples = p.sample(batchSize)
    mean = torch.cos(samples).mean(0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT.detach().numpy(), decimal=2)


def test_logProbImp():
    batchSize = 3
    nvars = [3, 16, 16]
    T = torch.tensor(4.0)
    B = 1.0

    mu = torch.randint(-10, 10, nvars).to(torch.float32) / 10 * 2.7
    kappa = (torch.arange(np.prod(nvars)) * 0.002).reshape(nvars).to(torch.float32) + 0.002

    p = source.VonMisesImp(nvars, mu, kappa, T=1.0)
    from torch.distributions import VonMises
    m = VonMises(mu, kappa)

    sampleb = m.sample([batchSize])
    samples = p.sample(batchSize)

    logPb = p.logProbability(sampleb)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples).reshape(batchSize, -1).sum(-1, keepdim=True)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)

    # test temperature
    p = source.VonMisesImp(nvars, mu, kappa, T=T)
    m = VonMises(mu, kappa/T)

    sampleb = m.sample([batchSize])
    samples = p.sample(batchSize)

    logPb = p.logProbability(sampleb)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples).reshape(batchSize, -1).sum(-1, keepdim=True)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)

    # test boundary
    p = source.VonMisesImp(nvars, mu, kappa, B=B, T=T)
    m = VonMises(mu, kappa/T)

    sampleb = m.sample([batchSize])
    samples = p.sample(batchSize)

    assert samples.max() <= B and samples.min() >= -B

    logPb = p.logProbability(sampleb / np.pi * B)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True) - np.log(B / np.pi) * np.prod(p.nvars)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples / B * np.pi).reshape(batchSize, -1).sum(-1, keepdim=True) - np.log(B / np.pi) * np.prod(p.nvars)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)

    # test T with batchSize
    p = source.VonMisesImp(nvars, mu, kappa, T=T.repeat(batchSize))
    m = VonMises(mu, kappa/T)

    sampleb = m.sample([batchSize])
    samples = p.sample(batchSize, T=T)

    logPb = p.logProbability(sampleb)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples).reshape(batchSize, -1).sum(-1, keepdim=True)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)


def test_logProb():
    batchSize = 100
    nvars = [3, 16, 16]
    T = torch.tensor(4.0)

    mu = torch.randint(-10, 10, nvars).to(torch.float32) / 10 * 2.7
    kappa = (torch.arange(np.prod(nvars)) * 0.002).reshape(nvars).to(torch.float32) + 0.002

    p = source.VonMises(nvars, mu, kappa, T=1.0)
    from torch.distributions import VonMises
    m = VonMises(mu, kappa)

    sampleb = m.sample([batchSize])
    samples = p.sample(batchSize)

    logPb = p.logProbability(sampleb)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples).reshape(batchSize, -1).sum(-1, keepdim=True)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)

    # test temperature
    p = source.VonMises(nvars, mu, kappa, T=T)
    m = VonMises(mu, kappa/T)

    sampleb = m.sample([batchSize])
    samples = p.sample(batchSize)

    logPb = p.logProbability(sampleb)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples).reshape(batchSize, -1).sum(-1, keepdim=True)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)

    # test T with batchSize
    p = source.VonMises(nvars, mu, kappa, T=T.repeat(batchSize))
    m = VonMises(mu, kappa/T)

    sampleb = m.sample([batchSize])
    samples = p.sample(batchSize, T=T)

    logPb = p.logProbability(sampleb)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples).reshape(batchSize, -1).sum(-1, keepdim=True)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)


if __name__ == "__main__":
    #test_sampleImp()
    test_logProbImp()