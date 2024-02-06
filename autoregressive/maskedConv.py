import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, maskType, dataChannels, augmentChannels=0, augmentOutput=True, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        assert maskType in ['A', 'B'], 'Invalid mask type.'

        outChannels, inChannels, height, width = self.weight.size()
        if augmentOutput:
            assert inChannels % (dataChannels + augmentChannels) == 0 and outChannels % (dataChannels + augmentChannels) == 0
        else:
            assert inChannels % (dataChannels + augmentChannels) == 0 and outChannels % dataChannels == 0
        yc, xc = height // 2, width // 2

        mask = torch.zeros(self.weight.size())
        mask[:, :, :yc, :] = 1
        mask[:, :, yc, :xc + 1] = 1

        if maskType == 'A':
            metaMask = torch.tril(torch.ones(dataChannels, dataChannels), diagonal=-1)
        else:
            metaMask = torch.tril(torch.ones(dataChannels, dataChannels))

        if augmentChannels > 0:
            if augmentOutput:
                metaMask = torch.cat([metaMask, torch.zeros(augmentChannels, dataChannels)], dim=0)
                metaMask = torch.cat([metaMask, torch.ones(augmentChannels + dataChannels, augmentChannels)], dim=1)
            else:
                metaMask = torch.cat([metaMask, torch.ones(dataChannels, augmentChannels)], dim=1)

        if augmentOutput:
            repeatOut = outChannels // (dataChannels + augmentChannels)
        else:
            repeatOut = outChannels // dataChannels
        repeatIn = inChannels // (dataChannels + augmentChannels)

        mask[:, :, yc, yc] = metaMask.repeat(repeatOut, repeatIn)

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(MaskedConv2d, self).forward(x)
        return x


class MaskedConv1d(nn.Conv1d):
    def __init__(self, *args, maskType, dataChannels, augmentChannels=0, augmentOutput=True, **kwargs):
        super(MaskedConv1d, self).__init__(*args, **kwargs)

        assert maskType in ['A', 'B'], 'Invalid mask type.'

        outChannels, inChannels, width = self.weight.size()
        if augmentOutput:
            assert inChannels % (dataChannels + augmentChannels) == 0 and outChannels % (dataChannels + augmentChannels) == 0
        else:
            assert inChannels % (dataChannels + augmentChannels) == 0 and outChannels % dataChannels == 0
        xc = width // 2

        mask = torch.zeros(self.weight.size())
        mask[:, :, :xc + 1] = 1

        if maskType == 'A':
            metaMask = torch.tril(torch.ones(dataChannels, dataChannels), diagonal=-1)
        else:
            metaMask = torch.tril(torch.ones(dataChannels, dataChannels))

        if augmentChannels > 0:
            if augmentOutput:
                metaMask = torch.cat([metaMask, torch.zeros(augmentChannels, dataChannels)], dim=0)
                metaMask = torch.cat([metaMask, torch.ones(augmentChannels + dataChannels, augmentChannels)], dim=1)
            else:
                metaMask = torch.cat([metaMask, torch.ones(dataChannels, augmentChannels)], dim=1)

        if augmentOutput:
            repeatOut = outChannels // (dataChannels + augmentChannels)
        else:
            repeatOut = outChannels // dataChannels
        repeatIn = inChannels // (dataChannels + augmentChannels)

        mask[:, :, xc] = metaMask.repeat(repeatOut, repeatIn)

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(MaskedConv1d, self).forward(x)
        return x


class MaskedResConv(nn.Module):
    def __init__(self, channel, kernelSize, hiddenChannels, hiddenConvLayers, hiddenKernelSize, hiddenWidth, hiddenFcLayers, category, augmentChannels=0,actFunc=[F.relu, nn.ReLU()]):
        super().__init__()

        self.channel = channel
        self.category = category

        self.firstConv = MaskedConv2d(channel + augmentChannels, 2 * hiddenChannels, kernelSize, padding=(kernelSize-1)//2, maskType= "A", dataChannels=channel, augmentChannels=augmentChannels)
        self.actFunc = actFunc
        hiddenConv = []
        for _ in range(hiddenConvLayers):
            hiddenConv.append(nn.Sequential(
                MaskedConv2d(2 * hiddenChannels, hiddenChannels, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels),
                self.actFunc[1],
                MaskedConv2d(hiddenChannels, hiddenChannels, hiddenKernelSize, padding=(hiddenKernelSize-1)//2, maskType="B", dataChannels=channel, augmentChannels=augmentChannels),
                self.actFunc[1],
                MaskedConv2d(hiddenChannels, 2 * hiddenChannels, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels)
            ))
        self.hiddenConv = nn.ModuleList(hiddenConv)

        self.firstFc = MaskedConv2d(2 * hiddenChannels, hiddenWidth, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels)

        hiddenFc = []
        for _ in range(hiddenFcLayers):
            hiddenFc.append(nn.Sequential(
                MaskedConv2d(hiddenWidth, hiddenWidth, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels)
            ))
        self.hiddenFc = nn.ModuleList(hiddenFc)

        self.finalFc = MaskedConv2d(hiddenWidth, category * channel, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels, augmentOutput=False)

    def forward(self, x):
        size = x.shape
        x = self.firstConv(x)
        x = self.actFunc[0](x)

        for layer in self.hiddenConv:
            tmp = x
            tmp = layer(tmp)
            # tmp = F.relu(tmp)
            tmp = self.actFunc[0](tmp)
            x = x + tmp

        x = self.firstFc(x)
        # x = F.relu(x)
        x = self.actFunc[0](x)

        for layer in self.hiddenFc:
            x = layer(x)
            # x = F.relu(x)
            x = self.actFunc[0](x)

        x = self.finalFc(x)

        return x.reshape(size[0], self.category, self.channel, size[-2], size[-1])


class MaskedResConv1d(nn.Module):
    def __init__(self, channel, kernelSize, hiddenChannels, hiddenConvLayers, hiddenKernelSize, hiddenWidth, hiddenFcLayers, category, augmentChannels=0):
        super().__init__()

        self.channel = channel
        self.category = category

        self.firstConv = MaskedConv1d(channel + augmentChannels, 2 * hiddenChannels, kernelSize, padding=(kernelSize-1)//2, maskType= "A", dataChannels=channel, augmentChannels=augmentChannels)

        hiddenConv = []
        for _ in range(hiddenConvLayers):
            hiddenConv.append(nn.Sequential(
                MaskedConv1d(2 * hiddenChannels, hiddenChannels, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels),
                nn.ReLU(),
                MaskedConv1d(hiddenChannels, hiddenChannels, hiddenKernelSize, padding=(hiddenKernelSize-1)//2, maskType="B", dataChannels=channel, augmentChannels=augmentChannels),
                nn.ReLU(),
                MaskedConv1d(hiddenChannels, 2 * hiddenChannels, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels)
            ))
        self.hiddenConv = nn.ModuleList(hiddenConv)

        self.firstFc = MaskedConv1d(2 * hiddenChannels, hiddenWidth, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels)

        hiddenFc = []
        for _ in range(hiddenFcLayers):
            hiddenFc.append(nn.Sequential(
                MaskedConv1d(hiddenWidth, hiddenWidth, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels)
            ))
        self.hiddenFc = nn.ModuleList(hiddenFc)

        self.finalFc = MaskedConv1d(hiddenWidth, category * channel, 1, maskType="B", dataChannels=channel, augmentChannels=augmentChannels, augmentOutput=False)

    def forward(self, x):
        size = x.shape
        x = self.firstConv(x)
        x = F.relu(x)

        for layer in self.hiddenConv:
            tmp = x
            tmp = layer(tmp)
            tmp = F.relu(tmp)
            x = x + tmp

        x = self.firstFc(x)
        x = F.relu(x)

        for layer in self.hiddenFc:
            x = layer(x)
            x = F.relu(x)

        x = self.finalFc(x)

        return x.reshape(size[0], self.category, self.channel, size[-1])
