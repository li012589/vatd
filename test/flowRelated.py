import torch
import numpy as np
from numpy.testing import assert_array_almost_equal,assert_array_equal

import os
import sys
sys.path.append(os.getcwd())
from utils import jacobian

def bijective(flow,batch=100,decimal=5, sampleMethod=None, args=[]):
    if sampleMethod is None:
        x,p = flow.sample(batch, *args)
    else:
        x,p = sampleMethod(batch)
    z,ip = flow.inverse(x, *args)
    xz,gp = flow.forward(z, *args)
    op = flow.prior.logProbability(z, *args)
    zx,ipp = flow.inverse(xz, *args)
    assert_array_almost_equal(x.detach().numpy(),xz.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(z.detach().numpy(),zx.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(ip.detach().numpy(),-gp.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(ip.detach().numpy(),ipp.detach().numpy(),decimal=decimal)
    if sampleMethod is None:
        assert_array_almost_equal(p.detach().numpy(),(op-gp).detach().numpy(),decimal=decimal)

def detJacobian(flow, batch=100, decimal=5, sampleMethod=None, inverse=False, compareLog=False, selectMask=None, args={}):
    if sampleMethod is None:
        z = flow.prior.sample(batch, *args).reshape(batch, -1).requires_grad_()
    else:
        z = sampleMethod(batch).reshape(batch, -1).requires_grad_()

    if inverse:
        x, ip = flow.inverse(z.reshape([batch] + flow.prior.nvars), *args)
    else:
        x, ip = flow.forward(z.reshape([batch] + flow.prior.nvars), *args)

    if selectMask is not None:
        x = torch.masked_select(x, selectMask)

    x = x.reshape(batch, -1)

    jac = jacobian(x, z)

    ipp = torch.det(jac)

    if compareLog:
        ipp = torch.log(ipp)
        ip = ip.reshape(-1)
    else:
        ip = torch.exp(ip.reshape(-1))

    assert_array_almost_equal(ipp.detach().numpy(), ip.detach().numpy(), decimal=decimal)


def temperatureProbRatio(flow, batch=100, decimal=1, T0=torch.tensor(.2), T=torch.tensor(1.0)):
    sample1, logProb1 = flow.sample(batch, T=T0)
    sample2, logProb2 = flow.sample(batch, T=T0)

    logProbT1 = flow.logProbability(sample1, T=T)
    logProbT2 = flow.logProbability(sample2, T=T)

    logRatioT0 = logProb1 - logProb2
    logRatioT = logProbT1 - logProbT2

    ratio = logRatioT / logRatioT0
    assert_array_almost_equal(ratio.reshape(batch).detach().numpy(), (T0/T).repeat(batch).detach().numpy(), decimal=decimal)


def saveload(flow,blankFlow,batch=100,decimal=5, sampleMethod=None, args={}):
    if sampleMethod is None:
        x,p = flow.sample(batch, *args)
    else:
        x,p = sampleMethod(batch)
    z,ip = flow.inverse(x, *args)
    d = flow.save()
    torch.save(d,"testsaving.saving")
    dd = torch.load("testsaving.saving")
    blankFlow.load(dd)
    op = blankFlow.prior.logProbability(z, *args)
    xz,gp = blankFlow.forward(z, *args)
    assert_array_almost_equal(x.detach().numpy(),xz.detach().numpy(),decimal=decimal)
    assert_array_almost_equal(ip.detach().numpy(),-gp.detach().numpy(),decimal=decimal)
    if sampleMethod is None:
        assert_array_almost_equal(p.detach().numpy(),(op-gp).detach().numpy(),decimal=decimal)

