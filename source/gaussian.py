import torch
import numpy as np

from .source import Source

class Gaussian(Source):
    def __init__(self, nvars, mu=None, logsigma=None, trainable=True, T=1.0, pixelProb=False, name="gaussian"):
        super(Gaussian,self).__init__(nvars, T, name)

        if logsigma is None:
            logsigma = torch.zeros(nvars)
        if mu is None:
            mu = torch.zeros(nvars)

        if pixelProb:
            self.mu = mu
            self.logsigma = logsigma
        else:
            self.mu = torch.nn.Parameter(mu.to(torch.float32), requires_grad=trainable)
            self.logsigma = torch.nn.Parameter(logsigma.to(torch.float32),requires_grad=trainable)

        self.pixelProb = pixelProb

    def sample(self, batchSize, T = None):
        if T is None:
            T = self.T
        size = [batchSize] + self.nvars
        return (torch.randn(size, dtype=self.logsigma.dtype).to(self.logsigma) * torch.exp(self.logsigma) * torch.sqrt(T) + self.mu)

    def energy(self, z):
        if self.pixelProb:
            return 0.5 * ((z - self.mu)**2 * torch.exp(-2 * self.logsigma))
        else:
            return 0.5 * ((z - self.mu)**2 * torch.exp(-2 * self.logsigma)).reshape(z.shape[0], -1).sum(dim = 1, keepdim=True)

    def logParitionFn(self, T=None):
        if T is None:
            T = self.T
        if len(T.shape) >= 1:
            T = T.reshape(-1, 1)
        if self.pixelProb:
            return 0.5 * np.log(2. * np.pi) + (self.logsigma + 0.5 * np.log(T))
        else:
            return 0.5 * np.prod(self.nvars) * np.log(2. * np.pi) + (self.logsigma.reshape(1, -1) + 0.5 * np.log(T)).sum(-1, keepdim=True)

class MultivariateGaussian(Source):
    def __init__(self, nvars, mu=None, logsigmaDiag=None, sigmaTria=None, trainable=True, T=1.0, name="MutliGaussian"):
        super(MultivariateGaussian, self).__init__(nvars, T, name)

        self.N = np.prod(nvars)
        if mu is None:
            mu = torch.zeros(self.N)
        if logsigmaDiag is None:
            logsigmaDiag = torch.zeros(self.N)
        if sigmaTria is None:
            sigmaTria = torch.zeros((self.N * (self.N - 1)) // 2)

        assert mu.shape[0] == self.N
        assert logsigmaDiag.shape[0] == self.N
        assert sigmaTria.shape[0] == (self.N * (self.N - 1)) // 2

        self.mu = torch.nn.Parameter(mu.to(torch.float32), requires_grad=trainable)
        self.logsigmaDiag = torch.nn.Parameter(logsigmaDiag.to(torch.float32), requires_grad=trainable)
        self.sigmaTria = torch.nn.Parameter(sigmaTria.to(torch.float32), requires_grad=trainable)

    def _sigma(self):
        sigma = torch.diag(torch.exp(self.logsigmaDiag))
        idx = torch.triu_indices(self.N, self.N, offset=1)
        sigma[idx[0], idx[1]] = self.sigmaTria
        return sigma

    def sample(self, batchSize, T=None):
        if T is None:
            T = self.T
        size = [batchSize] + [self.N]
        return (torch.randn(size, dtype=self.logsigmaDiag.dtype).to(self.logsigmaDiag).matmul(self._sigma()) * torch.sqrt(T) + self.mu).reshape(-1, *self.nvars)

    def energy(self, z):
        x = z.reshape(-1, self.N)
        assert z.shape[0] == x.shape[0]

        return 0.5 * ((x - self.mu).matmul(torch.cholesky_inverse(self._sigma().T)) * (x - self.mu)).sum(-1, keepdim=True)

    def logParitionFn(self, T=None):
        if T is None:
            T = self.T
        if len(T.shape) >= 1:
            T = T.reshape(-1, 1)
        return 0.5 * self.N * np.log(2. * np.pi).item()  + (self.logsigmaDiag.reshape(1, -1) + 0.5 * torch.log(T)).sum(-1, keepdim=True)