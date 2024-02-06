import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial

from .maskedConv import MaskedConv2d


class DiscretePixelCNN(nn.Module):
    def __init__(self, size, channel, maskedConv, fixFirst=None, mapping=None, reverseMapping=None, device=torch.device("cpu"), name="discretePixelCNN"):
        super(DiscretePixelCNN, self).__init__()

        self.size = size
        self.channel = channel

        self.maskedConv = maskedConv

        self.fixFirst = fixFirst
        self.mapping = mapping
        self.reverseMapping = reverseMapping

        self.name = name
        self.device = device

    def sample(self, batchSize, T=None):
        sample = torch.zeros(batchSize, self.channel, self.size[0], self.size[1]).to(self.device)
        if T is not None:
            if len(T.shape) >= 1:
                T = T.reshape(T.shape[0], *([1] * (len(sample.shape) - 1))).repeat(1, 1, self.size[0], self.size[1]).to(self.device)
            else:
                T = T.repeat(batchSize, 1, self.size[0], self.size[1]).to(self.device)

            sample = torch.cat([sample, T], dim=1)

        for i in range(self.size[0]):
            for j in range(self.size[1]):

                # fix the first element of the samples to be a fixed value
                if self.fixFirst is not None and i == 0 and j == 0:
                    if T is not None:
                        sample[:, :-1, 0, 0] = self.fixFirst
                    else:
                        sample[:, :, 0, 0] = self.fixFirst
                    continue

                for k in range(self.channel):
                    unnormalized = self.maskedConv.forward(sample)
                    sample[:, k, i, j] = torch.multinomial(torch.softmax(unnormalized[:, :, k, i, j], dim=1), 1).squeeze().float()
        if T is not None:
            sample = sample[:, :-1, :, :]

        if self.mapping is not None:
            sample = self.mapping(sample)

        return sample

    def logProbability(self, sample, T=None):
        if self.reverseMapping is not None:
            sample = self.reverseMapping(sample)

        if self.fixFirst is not None:
            assert (sample[:, :, 0, 0] == self.fixFirst).sum() == np.prod(sample.shape[:2])

        if T is not None:
            if len(T.shape) >= 1:
                T = T.reshape(T.shape[0], *([1] * (len(sample.shape) - 1))).repeat(1, 1, self.size[0], self.size[1]).to(self.device)
            else:
                T = T.repeat(sample.shape[0], 1, self.size[0], self.size[1]).to(self.device)

            sample = torch.cat([sample, T], dim=1)

        unnormalized = self.maskedConv.forward(sample)
        probability = torch.softmax(unnormalized, dim=1)

        if T is not None:
            sample = sample[:, :-1, :, :]

        logProbSelected = torch.log(probability.gather(1, sample.long().unsqueeze(1)))

        if self.fixFirst is not None:
            logProbSelected = logProbSelected.flatten(-2, -1).split([1, np.prod(self.size) - 1], -1)[1]

        return logProbSelected.view(logProbSelected.shape[0], -1).sum(1, keepdim=True)


class ContinuousPixelCNN(DiscretePixelCNN):
    def __init__(self, size, channel, maskedConv, fixFirst=None, mapping=None, reverseMapping=None, device=torch.device("cpu"), name="continuousPixelCNN"):
        super(ContinuousPixelCNN, self).__init__(size, channel, maskedConv, fixFirst, mapping, reverseMapping, device, name)

        self.prior = prior
        self.Nparam = Nparam
        self.Tparam = Tparam
        self.Tfunc = Tfunc

    def sample(self, batchSize, T=1):
        sample = torch.zeros(batchSize, self.channel, self.size[0], self.size[1]).to(self.device)

        for i in range(self.size[0]):
            for j in range(self.size[1]):

                # fix the first element of the samples to be a fixed value
                if self.fixFirst is not None and i == 0 and j == 0:
                    sample[:, :, 0, 0] = self.fixFirst
                    continue

                for k in range(self.channel):
                    params = self.maskedConv.forward(sample)
                    params = params[:, :, k, i, j]
                    dist = self.prior(*[params[:, n] if n != self.Tparam else self.Tfunc(params[:, n], T) for n in range(self.Nparam)])
                    sample[:, k, i, j] = dist.sample(1)

        if self.mapping is not None:
            sample = self.mapping(sample)

        return sample

    def logProbability(self, sample, T=1):
        if self.reverseMapping is not None:
            sample = self.reverseMapping(sample)

        params = self.maskedConv.forward(sample)
        dist = self.prior(*[params[:, n, :, :] if n != self.Tparam else self.Tfunc(params[:, n, :, :], T) for n in range(self.Nparam)])

        logProb = dist.logProbability(sample)

        if self.fixFirst is not None:
            assert (sample[:, :, 0, 0] == self.fixFirst).sum() == np.prod(sample.shape[:2])
            logProb[:, :, 0, 0] == 0

        return logProb.view(sample.shape[0], -1).sum(-1, keepdim=True)
