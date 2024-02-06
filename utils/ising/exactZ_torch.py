import torch
import numpy as np
import math


def h(j, beta):
    return beta * j


def h_star(j, beta):
    return torch.arctanh(torch.exp(-2 * beta * j))


def gamma(n, j, beta, r):
    return torch.arccosh(torch.cosh(2*h_star(j,beta)) * torch.cosh(2*h(j,beta)) -
                         torch.sinh(2*h_star(j,beta)) * torch.sinh(2*h(j,beta)) * np.cos(r*np.pi/n))


def logZ(n, j, beta):
    return torch.logsumexp(torch.cat([
        torch.cat([torch.log(2*torch.cosh(n/2*gamma(n,j,beta,2*r))).reshape(1) for r in range(n)], dim=0).sum(0, keepdim=True),
        torch.cat([torch.log(2*torch.sinh(n/2*gamma(n,j,beta,2*r))).reshape(1) for r in range(n)], dim=0).sum(0, keepdim=True),
        torch.cat([torch.log(2*torch.cosh(n/2*gamma(n,j,beta,2*r+1))).reshape(1) for r in range(n)], dim=0).sum(0, keepdim=True),
        torch.cat([torch.log(2*torch.sinh(n/2*gamma(n,j,beta,2*r+1))).reshape(1) for r in range(n)], dim=0).sum(0, keepdim=True),
    ], dim=0), dim=0) - torch.log(torch.tensor(2.)) + 1/2*n**2*torch.log(2*torch.sinh(2*h(j,beta)))


def freeEnergy(n, j, beta):
    return -1/n**2/beta * logZ(n,j,beta)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',type=int,default=4,help='L')
    parser.add_argument('-T',type=float,default=2.269185314213022,help='T')
    #parser.add_argument('-beta',type=float,default=1.0,help='beta')
    args = parser.parse_args()
    beta = torch.tensor(1/args.T, dtype=torch.float64)
    print ('#n:', args.n, 'beta:', beta)
    print(logZ(n=args.n, j=1, beta=beta).item())
    print(beta, freeEnergy(n=args.n, j=1, beta=beta).item())
