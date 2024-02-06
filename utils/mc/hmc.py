import numpy as np
import torch
from torch import nn
from torch.autograd import grad as torchgrad

def HMCwithAccept(energy, x, length, steps, epsilon, adaptiveAcptRate=None, adaptiveDecay=0.5, adaptiveConsiderRange=15):
    shape = [i if no==0 else 1 for no,i in enumerate(x.shape)]
    acptRate = []
    if adaptiveAcptRate is not None:
        totalTime = steps * epsilon
        originalEps = epsilon

    def grad(z):
        return torchgrad(energy(z),z,grad_outputs=z.new_ones(z.shape[0], 1))[0]

    torch.set_grad_enabled(False)
    E = energy(x)
    torch.set_grad_enabled(True)
    g = grad(x.requires_grad_())
    torch.set_grad_enabled(False)
    g = g.detach()
    for l in range(length):
        p = x.new_empty(size=x.size()).normal_()
        H = ((0.5*p*p).reshape(p.shape[0], -1).sum(dim=1, keepdim=True) + E)
        xnew = x
        gnew = g
        for s in range(steps):
            p = p - epsilon* gnew/2.
            xnew = (xnew + epsilon * p)
            torch.set_grad_enabled(True)
            gnew = grad(xnew.requires_grad_())
            torch.set_grad_enabled(False)
            xnew = xnew.detach()
            gnew = gnew.detach()
            p = p - epsilon* gnew/2.
        Enew = energy(xnew)
        Hnew = (0.5*p*p).reshape(p.shape[0], -1).sum(dim=1, keepdim=True) + Enew
        diff = H-Hnew
        accept = (diff.exp() >= diff.uniform_()).to(x)
        acptRate.append((accept.sum() / accept.shape[0]).item())
        accept = accept.reshape(-1).nonzero().view(-1)

        if adaptiveAcptRate is not None:
            if np.mean(acptRate[-adaptiveConsiderRange:]) <= adaptiveAcptRate:
                epsilon = epsilon * adaptiveDecay
            else:
                epsilon = epsilon / adaptiveDecay
            epsilon = min(epsilon, originalEps)

            steps = int(np.ceil(totalTime / epsilon))

        E[accept] = Enew[accept]
        x[accept] = xnew[accept]
        g[accept] = gnew[accept]

    torch.set_grad_enabled(True)

    return x, acptRate, [steps, epsilon]

def HMC(*args,**kwargs):
    x, acptRate, intParams = HMCwithAccept(*args,**kwargs)
    return x


class HMCsampler(nn.Module):
    def __init__(self,energy,nvars, epsilon=0.01, interSteps=10 , thermalSteps = 10):
        super(HMCsampler,self).__init__()
        self.nvars = nvars
        self.energy = energy
        self.interSteps = interSteps
        self.inital = HMC(self.energy,torch.randn(nvars),thermalSteps,interSteps)

    def step(self):
        self.inital = HMC(self.energy,self.inital,1,interSteps,epsilon)
        return self.inital