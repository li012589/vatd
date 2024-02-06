import torch
import math
import numpy as np

from .source import Source

from torch.distributions import VonMises as VM


class VonMisesImp(Source):
    '''
    element-wise von Mises distribution.

    Notations and formulates from [1] P. 49
    [1] N. I. Fisher, Statistical analysis of circular data, Cambridge University Press, (1993).
    '''
    def __init__(self, nvars, mu=None, kappa=None, B=np.pi, trainMu=True, trainKappa=False, kpClamp=4e-4, kappaMapper=None, pixelProb=False, T=1.0, name="vonMises"):
        super(VonMisesImp, self).__init__(nvars, T, name)

        if kappa is None:
            kappa = torch.ones(nvars)

        if mu is None:
            mu = torch.zeros(nvars)

        if kappaMapper is not None:
            assert isinstance(kappaMapper, torch.nn.modules.module.Module)
        self.kappaMapper = kappaMapper

        assert self.nvars == list(mu.shape)
        assert self.nvars == list(kappa.shape)

        if pixelProb:
            self.mu = mu
            self.kappa = kappa
        else:
            self.mu = torch.nn.Parameter(mu.to(torch.float32), requires_grad=trainMu)
            self.kappa = torch.nn.Parameter(kappa.to(torch.float32), requires_grad=trainKappa)
        self.B = B
        self.kpClamp = kpClamp
        self.pixelProb = pixelProb

        # placeholder for grad correction term
        self._logProb = None
        self._U = None
        self._R = None

    #@torch.no_grad()
    def sample(self, batchSize, T=None, eps=1e-7, saveFix=True):
        if T is None:
            T = self.T

        if len(T.shape) >= 1:
            T = T.reshape(T.shape[0], *([1] * len(self.nvars)))

        if not hasattr(self, "kappaMapper") or self.kappaMapper is None:
            kappa = torch.clamp(self.kappa/T, self.kpClamp)
        else:
            kappa = self.kappaMapper(self.kappa)/T

        a = 1.0 + torch.sqrt(1.0 + 4.0 * kappa**2)
        b = (a - torch.sqrt(2.0 * a)) / (2.0 * kappa)
        r = (1.0 + b**2) / (2.0 * b)
        self. _R = r

        if len(T.shape) >= 1:
            a, b, r, kp = [term.reshape(-1) for term in (a, b, r, kappa)]
        else:
            a, b, r, kp = [term.repeat(batchSize, *[1] * len(self.nvars)).reshape(-1) for term in (a, b, r, kappa)]

        U = torch.zeros([batchSize] + self.nvars).to(a).view(-1)
        theta = torch.zeros([batchSize] + self.nvars).to(a).view(-1)
        mask = torch.ones([batchSize] + self.nvars).to(a).bool().view(-1)

        while mask.sum() != 0:
            maskIdx = mask.nonzero().view(-1)

            u = torch.rand([mask.sum(), 3]).to(a)

            aGather, bGather, rGather, kpGather = [term[maskIdx] for term in (a, b, r, kp)]

            z = torch.cos(u[:, 0] * torch.pi)
            f = (1 + rGather * z) / (rGather + z)
            c = kpGather * (rGather - f)

            accept = (c * (2 - c) > u[:, 1]) | (torch.log(c / u[:, 1]) + 1 - c >= 0)
            acceptIdx = accept.reshape(-1).nonzero().view(-1)

            theta[maskIdx[acceptIdx]] = (u[acceptIdx, -1] - 0.5).sign() * torch.acos(torch.clamp(f[acceptIdx], -1+eps, 1-eps))
            mask[maskIdx[acceptIdx]] = 0
            U[maskIdx[acceptIdx]] = u[acceptIdx, 0]

        theta = (theta.reshape(batchSize, *self.nvars) + self.mu + torch.pi) % (2 * torch.pi) - torch.pi
        theta = theta / np.pi * self.B
        self._U = U.reshape(batchSize, *self.nvars)

        return theta

    def energy(self, z):
        if not hasattr(self, "kappaMapper") or self.kappaMapper is None:
            kappa = torch.clamp(self.kappa, self.kpClamp)
        else:
            kappa = self.kappaMapper(self.kappa)
        z = z / self.B * np.pi
        res = kappa * torch.cos(z - self.mu)
        if self.pixelProb:
            return -res
        else:
            return -res.reshape(res.shape[0], -1).sum(-1, keepdim=True)

    def logParitionFn(self, T=None):
        if T is None:
            T = self.T
        if not hasattr(self, "kappaMapper") or self.kappaMapper is None:
            kappa = torch.clamp(self.kappa, self.kpClamp)
        else:
            kappa = self.kappaMapper(self.kappa)
        if len(T.shape) >= 1:
            T = T.reshape(-1, 1)
        if self.pixelProb:
            kappa = kappa / T
            return np.log(2.0*torch.pi) + _log_modified_bessel_fn(kappa) + np.log(self.B / np.pi)
        else:
            kappa = kappa.reshape(1, -1) / T
            return np.log(2.0 * self.B) * np.prod(self.nvars) + _log_modified_bessel_fn(kappa).sum(-1, keepdim=True)

    def logProbability(self, x, T=None, saveFix=True):
        logProb = super().logProbability(x, T)
        if saveFix:
            self._logProb = logProb
        return logProb

    def logAceptRej(self):
        '''
        Used to reparameterize the accept-reject sampling, see Naesseth17

        !!! should fisrt sample samples and compute the logP from this distribution !!!
        '''
        assert self._logProb is not None
        assert self._U is not None
        assert self._R is not None

        logQ = self._logProb

        e = torch.pi * self._U
        cos = torch.cos(e)
        r = self._R

        logJaco = 0.5 * torch.log(r**2 - 1) + math.log(math.pi) - torch.log(r + cos)

        return logQ + logJaco.reshape(logQ.shape[0], -1).sum(-1, keepdim=True)


class VonMises(Source):
    def __init__(self, nvars, mu=None, kappa=None, B=np.pi, trainMu=True, trainKappa=False, T=1.0, name="vonMises"):
        super(VonMises, self).__init__(nvars, T, name)

        if kappa is None:
            kappa = torch.ones(nvars)

        if mu is None:
            mu = torch.zeros(nvars)

        self.mu = torch.nn.Parameter(mu.to(torch.float32), requires_grad=trainMu)
        self.kappa = torch.nn.Parameter(kappa.to(torch.float32), requires_grad=trainKappa)
        self.B = B


    def sample(self, batchSize, T=None):
        if T is None:
            T = self.T

        m = VM(self.mu, self.kappa/T)
        return m.sample([batchSize]) * self.B / np.pi

    def logProbability(self, z, T=None):
        if T is None:
            T = self.T

        if len(T.shape) >= 1:
            T = T.reshape(z.shape[0], 1)

        z = z / self.B * np.pi

        energyWithT = - (self.kappa * torch.cos(z - self.mu)).reshape(z.shape[0], -1).sum(-1, keepdim=True) / T
        logParitionFn = (math.log(2 * math.pi) + _log_modified_bessel_fn(self.kappa.reshape(1, -1) / T, order=0)).sum(-1, keepdim=True)

        return -energyWithT -logParitionFn - np.log(self.B / np.pi) * np.prod(self.nvars)


'''
The followings are from pytorch implementation of von Mises

https://pytorch.org/docs/stable/_modules/torch/distributions/von_mises.html#VonMises
https://github.com/pytorch/pytorch/blob/master/torch/distributions/von_mises.py
'''

def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_I0_COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                  -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]
_I1_COEF_SMALL = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.2658733e-1, 0.301532e-2, 0.32411e-3]
_I1_COEF_LARGE = [0.39894228, -0.3988024e-1, -0.362018e-2, 0.163801e-2, -0.1031555e-1,
                  0.2282967e-1, -0.2895312e-1, 0.1787654e-1, -0.420059e-2]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    """
    Returns ``log(I_order(x))`` for ``x > 0``,
    where `order` is either 0 or 1.
    """
    assert order == 0 or order == 1

    # compute small solution
    y = (x / 3.75)
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    result = torch.where(x < 3.75, small, large)
    return result

