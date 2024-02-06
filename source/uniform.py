import torch
import numpy as np

from .source import Source

class UniformImp(Source):
    def __init__(self, nvars, low, high, ignoreOutBound=False, outBoundE=1e3, dtype=torch.float32, T=1.0, name="unifrom"):
        super(UniformImp, self).__init__(nvars, T, name)

        if type(low) is not torch.tensor:
            low = torch.tensor(low)
        if type(high) is not torch.tensor:
            high = torch.tensor(high)
        self.ignoreOutBound = ignoreOutBound
        self.outBoundE = outBoundE
        self.low = torch.nn.Parameter(low, requires_grad=False)
        self.high = torch.nn.Parameter(high, requires_grad=False)
        self.dtype = dtype

    def sample(self, batchSize, T=None):
        return torch.rand([batchSize] + self.nvars, dtype=self.dtype).to(device=self.low.device) * (self.high - self.low) + self.low

    def energy(self, z):
        return self.energyWithT(z, None)

    def energyWithT(self, z, T=None):
        '''
        from torch.distributions.uniform.Uniform
        '''
        ob = 0
        if hasattr(self, 'ignoreOutBound') and self.ignoreOutBound:
            if self.outBoundE is not None:
                ob = self.low.gt(z).type_as(self.low) * self.high.le(z).type_as(self.low)
                ob = ob.reshape(z.shape[0], -1).sum(-1, keepdim=True) * self.outBoundE
            else:
                z = torch.clip(z, self.low, self.high)
                print("Out-of-boundary detected!")

        lb = self.low.le(z).type_as(self.low)
        ub = self.high.gt(z).type_as(self.low)
        return -torch.log(lb.mul(ub)).reshape(z.shape[0], -1).sum(-1, keepdim=True) + ob

    def logParitionFn(self, T=None):
        return torch.log(self.high - self.low) * np.prod(self.nvars)


class Uniform(Source):
    def __init__(self, nvars, low, high, ignoreOutBound=False, T=1.0, name="uniform"):
        super(Uniform, self).__init__(nvars, T, name)

        if type(low) is not torch.tensor:
            low = torch.tensor(low).float()
        if type(high) is not torch.tensor:
            high = torch.tensor(high).float()
        self.ignoreOutBound = ignoreOutBound
        self.low = torch.nn.Parameter(low, requires_grad=False)
        self.high = torch.nn.Parameter(high, requires_grad=False)
        self.dist = torch.distributions.Uniform(self.low, self.high)

    def sample(self, batchSize, T=None):
        return self.dist.sample([batchSize] + self.nvars)

    def logProbability(self, z, T=None):
        if hasattr(self, 'ignoreOutBound') and self.ignoreOutBound:
            z = torch.clip(z, self.low, self.high)
        return self.dist.log_prob(z).reshape(z.shape[0], -1).sum(-1, keepdim=True)

