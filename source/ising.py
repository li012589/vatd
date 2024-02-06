import numpy as np
import torch
import torch.nn.functional as F

import scipy.sparse as sps
from scipy.linalg import eigh, inv, det
from numpy import zeros
import math

from .source import Source
from utils import roll, isingLogz

class Lattice:
    def __init__(self,L, d, BC='periodic'):
        self.L = L
        self.d = d
        self.shape = [L]*d
        self.Nsite = L**d
        self.BC = BC

    def move(self, idx, d, shift):
        coord = self.index2coord(idx)
        coord[d] += shift

        if self.BC != 'periodic':
            if (coord[d]>=self.L) or (coord[d]<0):
                return None
        #wrap around because of the PBC
        if (coord[d]>=self.L): coord[d] -= self.L;
        if (coord[d]<0): coord[d] += self.L;

        return self.coord2index(coord)

    def index2coord(self, idx):
        coord = zeros(self.d, int)
        for d in range(self.d):
            coord[self.d-d-1] = idx%self.L;
            idx /= self.L
        return coord

    def coord2index(self, coord):
        idx = coord[0]
        for d in range(1, self.d):
            idx *= self.L;
            idx += coord[d]
        return idx

class Hypercube(Lattice):
    def __init__(self,L, d, BC='periodic'):
        super(Hypercube, self).__init__(L, d, BC)
        self.Adj = zeros((self.Nsite,self.Nsite), int)
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)

                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0

class ContinuousIsing(Source):
    def __init__(self, L, d, T, exact=True, eps=0.1, name = None):
        if name is None:
            name = "ContinuousIsing_l"+str(L)+"_d" +str(d)+"_t"+str(T)
        super(ContinuousIsing,self).__init__([L**d], T, name)
        self.lattice = Hypercube(L, d, 'periodic')
        try:
            device = T.device
        except:
            device = torch.device("cpu")
        N = torch.from_numpy(self.lattice.Adj).float().to(device)

        w, v = torch.linalg.eigh(N)
        offset = -w.min().detach().item()

        self.offset = offset / T + eps
        self.K = N / T + torch.eye(L**d).to(device)*(self.offset)

        _, logdet = torch.linalg.slogdet(self.K)

        self.logdet = logdet
        #print (sign)
        #print (0.5*self.nvars[0] *(np.log(4.)-offset - np.log(2.*np.pi)) - 0.5*logdet)
        Kinv = torch.inverse(self.K)
        self.register_buffer("Kinv",Kinv)

        self.isingLogz = isingLogz(n = L, j=1.0, beta = (1/self.T).item())
        self.exactloss = -self.isingLogz + 0.5*self.nvars[0] *(np.log(4.)- self.offset - np.log(2.*np.pi)) - 0.5*logdet.detach().cpu().numpy()
        if exact:
            print("Lower bound for loss:", self.exactloss)

    def energy(self, x):
        return -(-0.5*(torch.mm(x.reshape(-1, self.nvars[0]),self.Kinv) * x.reshape(-1, self.nvars[0])).sum(dim=1, keepdim=True) \
        + (torch.nn.Softplus()(2.*x.reshape(-1, self.nvars[0])) - x.reshape(-1, self.nvars[0]) - math.log(2.)).sum(dim=1, keepdim=True))

    def energyWithT(self, x, T):
        if T != self.T:
            raise Exception("Only T initalized (T="+str(self.T)+") is supported")
        return self.energy(x)


class VanillaIsing(Source):
    def __init__(self, L, d, T, breakSym=True, exact=True, name = None):
        if name is None:
            name = "ContinuousIsing_l"+str(L)+"_d" +str(d)+"_t"+str(T)
        super(VanillaIsing,self).__init__([L**d], T, name)
        self.lattice = Hypercube(L, d, 'periodic')
        try:
            device = T.device
        except:
            device = torch.device("cpu")
        N = torch.from_numpy(self.lattice.Adj).float().to(device) * -1
        self.register_buffer("N", N)
        self.breakSym = breakSym
        if exact:
            self.exactloss = isingLogz(n = L, j=1.0, beta = (1/self.T).item())
            print("Lower bound for loss:", self.exactloss)

    def energy(self, x):
        if self.breakSym:
            # fix the first spin to be up
            assert (x[:, 0, 0, 0] == 1).sum() == x.shape[0]

        return ((x.flatten(-2) @ self.N) * x.flatten(-2)).sum([-2, -1]).unsqueeze(-1)/2
