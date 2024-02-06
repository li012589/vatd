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

def test_gaussianSample():
    batchSize = 100000
    nvars = [2, 8, 8]
    T = torch.tensor(4.0)
    mean = torch.arange(np.prod(nvars)).reshape(nvars).to(torch.float32)
    logsigma = (torch.arange(np.prod(nvars)) * 0.01).reshape(nvars).to(torch.float32)

    p = source.Gaussian(nvars, mu=mean, logsigma=logsigma, T=1.0)
    samples = p.sample(batchSize)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(samples.mean(0).detach().numpy(), mean, decimal=1)
    assert_almost_equal(torch.log(samples.std(0)).detach().numpy(), logsigma, decimal=1)

    # test temperature

    samples = p.sample(batchSize, T=T)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(samples.mean(0).detach().numpy(), mean, decimal=1)
    assert_almost_equal(torch.log(samples.std(0)).detach().numpy(), logsigma + 0.5 * np.log(T), decimal=1)

    p = source.Gaussian(nvars, mu=mean, logsigma=logsigma, T=T.item())
    samples = p.sample(batchSize)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(samples.mean(0).detach().numpy(), mean, decimal=1)
    assert_almost_equal(torch.log(samples.std(0)).detach().numpy(), logsigma + 0.5 * np.log(T), decimal=1)


def test_gaussianLogProbab():
    batchSize = 100
    nvars = [3, 16, 16]
    T = torch.tensor(4.0)
    mean = torch.randint(-10, 10, nvars).to(torch.float32)
    logsigma = (torch.arange(np.prod(nvars)) * 0.002).reshape(nvars).to(torch.float32)

    p = source.Gaussian(nvars, mu=mean, logsigma=logsigma, T=1.0)

    def normalizer(logsigma=logsigma):
        return 0.5 * np.prod(nvars)* np.log((2*np.pi)) + logsigma.sum()

    def logE(x, logsigma=logsigma):
        return (-1/2 * ((x - mean)**2 * torch.exp(-2 * logsigma)).reshape(batchSize, -1).sum(-1))

    samplesLow = torch.randn([batchSize] + nvars) * 10 + 3
    samplesHigh = p.sample(batchSize)

    logPLow = p.logProbability(samplesLow)
    logPHigh = p.logProbability(samplesHigh)

    norm = p.logParitionFn()

    assert_almost_equal(norm.detach().numpy(), normalizer().detach().numpy())

    _logPLow = logE(samplesLow) - normalizer()
    _logPHigh = logE(samplesHigh) - normalizer()

    assert_array_equal(logPLow.shape, [batchSize, 1])
    assert_array_equal(logPHigh.shape, [batchSize, 1])

    assert_almost_equal(logPHigh.detach().numpy().reshape(-1), _logPHigh.detach().numpy())
    assert_almost_equal(logPLow.detach().numpy().reshape(-1), _logPLow.detach().numpy())

    # test with different T

    samples = p.sample(batchSize)
    logP = p.logProbability(samples, T=T)
    norm = p.logParitionFn(T = T)

    assert_almost_equal(norm.detach().numpy(), normalizer(logsigma = logsigma + 0.5 * np.log(T)).detach().numpy())

    _logP = logE(samples, logsigma=logsigma + 0.5 * np.log(T)) - normalizer(logsigma = logsigma + 0.5 * np.log(T))

    assert_array_equal(logP.shape, [batchSize, 1])
    assert_almost_equal(logP.detach().numpy().reshape(-1), _logP.detach().numpy(), decimal=4)

    # test T with batchSize

    samples = p.sample(batchSize)
    logP = p.logProbability(samples, T=T.repeat(batchSize))
    norm = p.logParitionFn(T = T.repeat(batchSize))

    assert_almost_equal(norm.detach().numpy(), normalizer(logsigma = logsigma + 0.5 * np.log(T)).detach().numpy())

    _logP = logE(samples, logsigma=logsigma + 0.5 * np.log(T)) - normalizer(logsigma = logsigma + 0.5 * np.log(T))

    assert_array_equal(logP.shape, [batchSize, 1])
    assert_almost_equal(logP.detach().numpy().reshape(-1), _logP.detach().numpy(), decimal=4)


def test_multivariateGaussianSample():
    batchSize = 10000000
    nvars = [2, 2, 3]
    N = np.prod(nvars)
    T = torch.tensor(4.0)
    mean = torch.arange(N).to(torch.float32)
    logsigmaDiag = (torch.log(torch.randint(2, 4, [N]))).to(torch.float32)
    #sigmaTria = torch.zeros([(N-1)*N//2]).to(torch.float32)
    sigmaTria = torch.randint(0, 2, [(N-1)*N//2]).to(torch.float32)

    sigmaMat = torch.diag(torch.exp(logsigmaDiag))
    idx = torch.triu_indices(N, N, offset=1)
    sigmaMat[idx[0], idx[1]] = sigmaTria
    sigmaMat = sigmaMat.T
    SIG = sigmaMat.matmul(sigmaMat.T)
    p = source.MultivariateGaussian(nvars, mu=mean, logsigmaDiag=logsigmaDiag, sigmaTria=sigmaTria, T=1.0)
    samples = p.sample(batchSize)
    covMat = samples.reshape(batchSize, -1).T.cov(correction=0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(samples.mean(0).detach().numpy(), mean.reshape(nvars), decimal=1)
    assert_almost_equal(covMat.detach().numpy(), SIG.detach().numpy(), decimal=1)

    # test temperature

    samples = p.sample(batchSize, T=T)
    covMat = samples.reshape(batchSize, -1).T.cov(correction=0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(samples.mean(0).detach().numpy(), mean.reshape(nvars), decimal=1)
    assert_almost_equal(covMat.detach().numpy(), T * SIG.detach().numpy(), decimal=1)

    p = source.MultivariateGaussian(nvars, mu=mean, logsigmaDiag=logsigmaDiag, sigmaTria=sigmaTria, T=T.item())
    samples = p.sample(batchSize)
    covMat = samples.reshape(batchSize, -1).T.cov(correction=0)

    assert_array_equal(samples.shape, [batchSize] + nvars)
    assert_almost_equal(samples.mean(0).detach().numpy(), mean.reshape(nvars), decimal=1)
    assert_almost_equal(covMat.detach().numpy(), T * SIG.detach().numpy(), decimal=1)


def test_multivariateGaussianProbab():
    batchSize = 100
    nvars = [2, 4, 4]
    N = np.prod(nvars)
    T = torch.tensor(4.0)
    mean = torch.arange(N).to(torch.float32)
    logsigmaDiag = (torch.log(torch.randint(1, 4, [N]))).to(torch.float32)
    #sigmaTria = torch.zeros([(N-1)*N//2]).to(torch.float32)
    sigmaTria = torch.randint(0, 2, [(N-1)*N//2]).to(torch.float32)

    sigmaMat = torch.diag(torch.exp(logsigmaDiag))
    idx = torch.triu_indices(N, N, offset=1)
    sigmaMat[idx[0], idx[1]] = sigmaTria
    sigmaMat = sigmaMat.T
    SIG = sigmaMat.matmul(sigmaMat.T)
    SIGinv = torch.cholesky_inverse(sigmaMat)
    p = source.MultivariateGaussian(nvars, mu=mean, logsigmaDiag=logsigmaDiag, sigmaTria=sigmaTria, T=1.0)

    samplesLow = torch.randn([batchSize] + nvars) * 10 + 3
    samplesHigh = p.sample(batchSize)

    def normalizer(logsigmaDiag=logsigmaDiag):
        return 0.5 * N * np.log((2*np.pi)) + logsigmaDiag.sum()

    def logE(x, SIGinv=SIGinv):
        return -1/2 * ((x - mean).matmul(SIGinv) * (x - mean)).sum(-1)

    logPLow = p.logProbability(samplesLow)
    logPHigh = p.logProbability(samplesHigh)

    norm = p.logParitionFn()

    assert_almost_equal(norm.detach().numpy(), normalizer().detach().numpy())

    samplesLow = samplesLow.reshape(batchSize, -1)
    samplesHigh = samplesHigh.reshape(batchSize, -1)

    _logPLow = logE(samplesLow) - normalizer()
    _logPHigh = logE(samplesHigh) - normalizer()

    assert_array_equal(logPLow.shape, [batchSize, 1])
    assert_array_equal(logPHigh.shape, [batchSize, 1])

    assert_almost_equal(logPHigh.detach().numpy().reshape(-1), _logPHigh.detach().numpy())
    assert_almost_equal(logPLow.detach().numpy().reshape(-1), _logPLow.detach().numpy())

    # test with different T

    samples = p.sample(batchSize)
    logP = p.logProbability(samples, T=T)
    norm = p.logParitionFn(T = T)

    assert_almost_equal(norm.detach().numpy(), normalizer(logsigmaDiag = logsigmaDiag + 0.5 * np.log(T)).detach().numpy())

    samples = samples.reshape(batchSize, -1)

    _logP = logE(samples, SIGinv=SIGinv/T) - normalizer(logsigmaDiag = logsigmaDiag + 0.5 * np.log(T))

    assert_array_equal(logP.shape, [batchSize, 1])
    assert_almost_equal(logP.detach().numpy().reshape(-1), _logP.detach().numpy())

    # test T with batchSize

    samples = p.sample(batchSize)
    logP = p.logProbability(samples, T=T.repeat(batchSize))
    norm = p.logParitionFn(T = T.repeat(batchSize))

    assert_almost_equal(norm.detach().numpy(), normalizer(logsigmaDiag = logsigmaDiag + 0.5 * np.log(T)).detach().numpy())

    samples = samples.reshape(batchSize, -1)

    _logP = logE(samples, SIGinv=SIGinv/T) - normalizer(logsigmaDiag = logsigmaDiag + 0.5 * np.log(T))

    assert_array_equal(logP.shape, [batchSize, 1])
    assert_almost_equal(logP.detach().numpy().reshape(-1), _logP.detach().numpy())


if __name__ == "__main__":
    # test_gaussianSample()
    #test_gaussianLogProbab()
    # test_multivariateGaussianSample()
    test_multivariateGaussianProbab()
