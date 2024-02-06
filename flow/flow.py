
import numpy as np
import torch
from torch import nn

class Flow(nn.Module):

    def __init__(self, prior = None,name = "Flow"):
        super(Flow, self).__init__()
        self.name = name
        self.prior = prior

    def __call__(self,*args,**kargs):
        return self.sample(*args,**kargs)

    def sample(self,batchSize, T=None, prior=None):
        if prior is None:
            prior = self.prior
        assert prior is not None
        if T is None:
            z = prior.sample(batchSize)
            logp = prior.logProbability(z)
            x, logp_ = self.forward(z)
        else:
            z = prior.sample(batchSize, T=T)
            logp = prior.logProbability(z, T=T)
            x, logp_ = self.forward(z, T=T)
        return x,logp-logp_

    def logProbability(self,x, T=None):
        if T is None:
            z, logp = self.inverse(x)
        else:
            z, logp = self.inverse(x, T=T)
        if self.prior is not None:
            return self.prior.logProbability(z, T) + logp
        return logp

    def energy(self, x):
        z, logp = self.inverse(x)
        return self.prior.energy(z) + logp

    def forward(self,x):
        raise NotImplementedError(str(type(self)))

    def inverse(self,z):
        raise NotImplementedError(str(type(self)))

    def transMatrix(self,sign):
        raise NotImplementedError(str(type(self)))

    def save(self):
        return self.state_dict()

    def load(self,saveDict):
        self.load_state_dict(saveDict)
        return saveDict