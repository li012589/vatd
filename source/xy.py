from .ising import Hypercube
from .source import Source

import torch
import numpy as np


class XYbreakSym(Source):
    '''
    XY model with the first spin also up (phi_0=0)
    '''
    def __init__(self, L, d, T=1.0, J=1.0, mappingBoundary=None, flg2d=False, fixFirst=0., name=None):
        if name is None:
            name = "XY_breakSym_l" + str(L) + "_d" + str(d) + "_J" + str(J)
        super(XYbreakSym, self).__init__([L**d - 1], T, name)
        N = torch.from_numpy(Hypercube(L, d, 'periodic').Adj).float() * -J
        self.register_buffer('N', N)
        self.mappingBoundary = mappingBoundary
        self.flg2d = flg2d
        self.fixFirst = fixFirst

    def energy(self, x):
        if self.mappingBoundary is not None:
            # map back to [-pi, pi], as energy fn is unnormalized, the det will cancel.
            x = x / (self.mappingBoundary) * np.pi
        if self.flg2d:
            assert (x[:, :, 0, 0] == self.fixFirst).sum() == np.prod(x.shape[:2])
        else:
            x = torch.cat([self.fixFirst * torch.ones(*x.shape[:-1], 1).to(x), x], dim=-1)
        outBound = ((x < -np.pi) * (x > np.pi)).sum()
        assert outBound == 0
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=1)
        return ((x.flatten(-2) @ self.N) * x.flatten(-2)).sum([-2, -1]).unsqueeze(-1)/2


class XY(Source):
    def __init__(self, L, d, T, J=1.0, name=None):
        if name is None:
            name = "XY_l" + str(L) + "_d" + str(d) + "_J" + str(J)

        super(XY, self).__init__([L**d], T, name)
        N = torch.from_numpy(Hypercube(L, d, 'periodic').Adj).float() * -J
        self.register_buffer('N', N)

    def energy(self, x):
        outBound = ((x < -np.pi) * (x > np.pi)).sum()
        assert outBound == 0
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=1)
        return ((x.flatten(-2) @ self.N) * x.flatten(-2)).sum([-2, -1]).unsqueeze(-1)/2
