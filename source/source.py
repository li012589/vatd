import numpy as np
import torch
from torch import nn
from utils import HMC, Metropolis

class Source(nn.Module):

    def __init__(self, nvars, T=1.0, name = "Flow"):
        super(Source, self).__init__()
        self.name = name
        self.nvars = nvars
        if type(T) is torch.Tensor:
            self.T = T
        else:
            self.T = torch.nn.Parameter(torch.tensor(T, dtype=torch.float32),requires_grad=False)

    def __call__(self,*args,**kargs):
        return self.sample(*args,**kargs)

    def sample(self, batchSize, T=None, thermalSteps = 50, interSteps=5, epsilon=0.1):
        if T is None:
            return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)
        else:
            '''
            TODO: mc with K changable
            '''
            return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)

    def logProbability(self, x, T=None):
        if T is None:
            T = self.T
        return -self.energyWithT(x, T) - self.logParitionFn(T = T)

    def energyWithT(self, x, T=None):
        if T is None:
            T = self.T
        if len(T.shape) >= 1:
            T = T.reshape(x.shape[0], 1)
        return self.energy(x) / T

    def energy(self, x):
        raise NotImplementedError(str(type(self)))

    def logParitionFn(self, T):
        raise NotImplementedError(str(type(self)))

    def save(self):
        return self.state_dict()

    def load(self,saveDict):
        self.load_state_dict(saveDict)
        return saveDict

    def _sampleWithHMC(self,batchSize,thermalSteps = 50, interSteps = 5, epsilon=0.1):
        try:
            tmp = next(self.parameters())
            device = tmp.device
            dtype = tmp.dtype
        except:
            device = self.device
            dtype = self.dtype
        inital = torch.randn([batchSize]+self.nvars,requires_grad=True).to(dtype).to(device)
        inital = HMC(self.energy,inital,thermalSteps,interSteps,epsilon)
        return inital.detach()

    def _sampleWithMetropolis(self,batchSize,thermalSteps = 100,tranCore = None):
        try:
            tmp = next(self.parameters())
            device = tmp.device
            dtype = tmp.dtype
        except:
            device = self.device
            dtype = self.dtype
        inital = torch.randn([batchSize]+self.nvars,requires_grad=True).to(dtype).to(device)
        inital = Metropolis(self.energy,inital,thermalSteps,tranCore)
        return inital.detach()