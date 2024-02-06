from .ising import Hypercube
from .source import Source

import torch
import numpy as np


class Phi4(Source):
    def __init__(self, L, d, T, t=1.0, mu=-198, lbd=25, name=None):
        if name is None:
            name = "Phi4_l"+str(L)+"_d"+str(d)+"_t"+str(t)+"_mu"+str(mu)+"_lbd"+str(lbd)

        super(Phi4, self).__init__([L**d], T, name)
        self.N = torch.from_numpy(Hypercube(L, d, 'periodic').Adj).float() * -t
                 #+ torch.diag(torch.tensor([mu] * (L**d)).float())
        self.t = t
        self.mu = mu
        self.lbd = lbd

    def energy(self, x):
        part1 = ((x.flatten(-2) @ self.N) * x.flatten(-2)).sum([-2, -1]).unsqueeze(-1)
        x2 = x**2
        part2 = (x2.sum([1, 2, 3]) * self.mu).unsqueeze(-1)
        part3 = ((x2**2).sum([1,2,3]) * self.lbd).unsqueeze(-1)
        return part1 + part2 + part3