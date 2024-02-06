import torch
from torch import nn
import numpy as np

from .source import Source


class ConcatenatedSource(Source):
    def __init__(self, sourceList, dim=-1, reshape=None, directLogProb=False, T=1.0, name="concatenatedSource"):
        nvars = sourceList[0].nvars.copy()
        for s in sourceList[1:]:
            nvars[dim] += s.nvars[dim]
        if reshape is not None:
            assert np.prod(reshape) == np.prod(nvars)
            self.original = nvars
            nvars = reshape
        else:
            self.original = None
        super().__init__(nvars, T=T, name=name)

        self.dim = -1
        self.sourceList = nn.ModuleList(sourceList)

        self.directLogProb = directLogProb

    def sample(self, batchSize, T=None):
        sampleList = []
        for s in self.sourceList:
            sampleList.append(s.sample(batchSize, T=T))
        res = torch.cat(sampleList, dim=self.dim)
        if self.original is not None:
            res = res.reshape(batchSize, *self.nvars)
        return res

    def energy(self, z):
        if self.original is not None:
            z = z.reshape(-1, *self.original)

        zparts = torch.split(z, [s.nvars[self.dim] for s in self.sourceList], dim=self.dim)
        energy = 0
        for no, s in enumerate(self.sourceList):
            energy += s.energy(zparts[no])

        return energy

    def logParitionFn(self, T=None):
        logParition = 0
        for s in self.sourceList:
            logParition += s.logParitionFn(T=T)
        return logParition

    def logProbability(self, z, T=None):
        if hasattr(self, 'directLogProb') and self.directLogProb:
            if self.original is not None:
                z = z.reshape(-1, *self.original)

            zparts = torch.split(z, [s.nvars[self.dim] for s in self.sourceList], dim=self.dim)
            logProb = 0
            for no, s in enumerate(self.sourceList):
                logProb += s.logProbability(zparts[no], T=T)
            return logProb
        else:
            return super().logProbability(z, T=T)
