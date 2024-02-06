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


def uniformMean(low, high):
    return (low + high) / 2


def uniformStd(low, high):
    return (high - low) / 12**0.5


def test_sampleImp():
    batchSize = 100000
    nvars = [2, 8, 8]
    T = torch.tensor(4.0)

    low = -3
    high = np.pi

    p = source.UniformImp(nvars, low, high, T=1.0)

    meanT = uniformMean(low, high)
    stdT = uniformStd(low, high)

    samples = p.sample(batchSize)
    mean = samples.mean()
    std = samples.std()

    assert samples.min() >= low
    assert samples.max() < high
    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT, decimal=2)
    assert_almost_equal(std.detach().numpy(), stdT, decimal=2)

    # test temperature
    p = source.UniformImp(nvars, low, high, T=T)

    meanT = uniformMean(low, high)
    stdT = uniformStd(low, high)

    samples = p.sample(batchSize)
    mean = samples.mean()
    std = samples.std()

    assert samples.min() >= low
    assert samples.max() < high
    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT, decimal=2)
    assert_almost_equal(std.detach().numpy(), stdT, decimal=2)


def test_logProbImp():
    batchSize = 100
    nvars = [3, 16, 16]
    T = torch.tensor(4.0)

    low = -3
    high = np.pi

    p = source.UniformImp(nvars, low, high, T=1.0)
    from torch.distributions import Uniform
    m = Uniform(low, high)

    sampleb = m.sample([batchSize] + nvars)
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
    p = source.UniformImp(nvars, low, high, T=T)
    m = Uniform(low, high)

    sampleb = m.sample([batchSize] + nvars)
    samples = p.sample(batchSize)

    logPb = p.logProbability(sampleb)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples).reshape(batchSize, -1).sum(-1, keepdim=True)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)


def test_sample():
    batchSize = 100000
    nvars = [2, 8, 8]
    T = torch.tensor(4.0)

    low = -3
    high = np.pi

    p = source.Uniform(nvars, low, high, T=1.0)

    meanT = uniformMean(low, high)
    stdT = uniformStd(low, high)

    samples = p.sample(batchSize)
    mean = samples.mean()
    std = samples.std()

    assert samples.min() >= low
    assert samples.max() < high
    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT, decimal=2)
    assert_almost_equal(std.detach().numpy(), stdT, decimal=2)

    # test temperature
    p = source.Uniform(nvars, low, high, T=T)

    meanT = uniformMean(low, high)
    stdT = uniformStd(low, high)

    samples = p.sample(batchSize)
    mean = samples.mean()
    std = samples.std()

    assert samples.min() >= low
    assert samples.max() < high
    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(mean.detach().numpy(), meanT, decimal=2)
    assert_almost_equal(std.detach().numpy(), stdT, decimal=2)


def test_logProb():
    batchSize = 100
    nvars = [3, 16, 16]
    T = torch.tensor(4.0)

    low = -3
    high = np.pi

    p = source.Uniform(nvars, low, high, T=1.0)
    from torch.distributions import Uniform
    m = Uniform(low, high)

    sampleb = m.sample([batchSize] + nvars)
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
    p = source.Uniform(nvars, low, high, T=T)
    m = Uniform(low, high)

    sampleb = m.sample([batchSize] + nvars)
    samples = p.sample(batchSize)

    logPb = p.logProbability(sampleb)
    _logPb = m.log_prob(sampleb).reshape(batchSize, -1).sum(-1, keepdim=True)

    logP = p.logProbability(samples)
    _logP = m.log_prob(samples).reshape(batchSize, -1).sum(-1, keepdim=True)

    assert_array_equal(logPb.shape, [batchSize, 1])
    assert_array_equal(logP.shape, [batchSize, 1])

    assert np.allclose(logPb.detach().numpy(), _logPb.detach().numpy(), atol=0, rtol=1e-5)
    assert np.allclose(logP.detach().numpy(), _logP.detach().numpy(), atol=0, rtol=1e-5)


if __name__ == "__main__":
    # test_sampleImp()
    # test_sample()
    # test_logProbImp()
    test_logProb()
