import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DecoratedResNet(nn.Module):
    def __init__(self, L, channel, kernelSize, outputChannels, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ReLU(), augmentChannels=None, fixFirst=None):
        super(DecoratedResNet, self).__init__()
        self.L = L
        self.firstConv = nn.Conv2d(channel, 2 * hiddenChannels, kernelSize, padding=(kernelSize - 1) // 2)

        self.hiddenConv = []
        for _ in range(hiddenConvLayers):
            self.hiddenConv.append(nn.Sequential(
                nn.Conv2d(2 * hiddenChannels, hiddenChannels, 1),
                activation,
                nn.Conv2d(hiddenChannels, hiddenChannels, kernelSize, padding=(kernelSize - 1) // 2),
                activation,
                nn.Conv2d(hiddenChannels, 2 * hiddenChannels, 1),
            ))
        self.hiddenConv = nn.ModuleList(self.hiddenConv)

        self.firstFc = nn.Conv2d(2 * hiddenChannels, hiddenWidth, 1)
        self.hiddenFc = []
        for _ in range(hiddenFcLayers):
            self.hiddenFc.append(
                nn.Conv2d(hiddenWidth, hiddenWidth, 1)
            )
        self.hiddenFc = nn.ModuleList(self.hiddenFc)

        self.finalFc = nn.Conv2d(hiddenWidth, outputChannels, 1)

        self.register_buffer('fixFirst', fixFirst)
        self.activation = activation
        self.augmentChannels = augmentChannels

    def _networkFwd(self, x):
        x = self.firstConv(x)
        x = self.activation(x)

        for layer in self.hiddenConv:
            tmp = x
            tmp = layer(tmp)
            tmp = self.activation(tmp)
            x = x + tmp

        x = self.firstFc(x)
        x = self.activation(x)

        for layer in self.hiddenFc:
            x = layer(x)
            x = self.activation(x)

        x = self.finalFc(x)
        return x

    def forward(self, inputs):
        if self.fixFirst is not None:
            if self.augmentChannels is not None:
                assert torch.allclose(inputs[:, :-self.augmentChannels, 0, 0], self.fixFirst)
            else:
                assert torch.allclose(inputs[:, :, 0, 0], self.fixFirst)
        inputs = self._networkFwd(inputs)
        return inputs


class DecoratedResMLP(nn.Module):
    def __init__(self, inSize, hiddenSize, hiddenMLPlayers, fcSize, fcLayers, outSize, activation=nn.ReLU(), reshape=None):
        super().__init__()
        self.firstMLP = nn.Linear(inSize, 2 * hiddenSize)

        self.hiddenMLP = []
        for _ in range(hiddenMLPlayers):
            self.hiddenMLP.append(nn.Sequential(
                nn.Linear(2 * hiddenSize, hiddenSize),
                activation,
                nn.Linear(hiddenSize, hiddenSize),
                activation,
                nn.Linear(hiddenSize, 2 * hiddenSize),
            ))
        self.hiddenMLP = nn.ModuleList(self.hiddenMLP)

        self.firstFc = nn.Linear(2 * hiddenSize, fcSize)
        self.hiddenFc = []
        for _ in range(fcLayers):
            self.hiddenFc.append(
                nn.Linear(fcSize, fcSize)
            )
        self.hiddenFc = nn.ModuleList(self.hiddenFc)

        self.finalFc = nn.Linear(fcSize, outSize)

        self.activation = activation
        self.reshape = reshape

    def _networkFwd(self, x):
        if self.reshape is not None:
            x = x.reshape(x.shape[0], -1)
        x = self.firstMLP(x)
        x = self.activation(x)

        for layer in self.hiddenMLP:
            tmp = x
            tmp = layer(tmp)
            tmp = self.activation(tmp)
            x = x + tmp

        x = self.firstFc(x)
        x = self.activation(x)

        for layer in self.hiddenFc:
            x = layer(x)
            x = self.activation(x)

        x = self.finalFc(x)
        if self.reshape is not None:
            x = x.reshape(x.shape[0], *self.reshape)
        return x

    def forward(self, inputs):
        inputs = self._networkFwd(inputs)
        return inputs


class MaskedMLPforSplineFlow(nn.Linear):
    def __init__(self, inSize, outSize, K, outFactor, maskType, reshape=None, device=None, dtype=None):
        super().__init__(inSize, outSize, device=device, dtype=dtype)

        assert maskType in ['A', 'B'], 'Invalid mask type.'

        self.reshape = reshape
        ratio = K / outFactor
        outWDim = int(ratio * outSize)
        if maskType == "B":
            inWDim = int(ratio * inSize)
        else:
            inWDim = int(inSize / 2)

        mask = torch.ones(outSize, inSize)
        mask[:outWDim, inWDim:] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        self.weight.data *= self.mask
        x = super().forward(x)
        if self.reshape is not None:
            x = x.reshape(x.shape[0], *self.reshape)
        return x


class SimpleMaskedMLPforSplineFlow(nn.Module):
    def __init__(self, dataShape, hiddenDims, K, outFactor, activation=None):
        super().__init__()
        if activation is None:
            activation = [nn.ReLU() for _ in range(len(hiddenDims))]

        assert len(activation) == len(hiddenDims) + 1

        layerList = []
        for no in range(len(activation)):
            if no == 0:
                layerList.append(MaskedMLPforSplineFlow(2 * np.prod(dataShape), hiddenDims[no], K=K, outFactor=outFactor, maskType="A"))
            elif no == len(activation) - 1:
                layerList.append(MaskedMLPforSplineFlow(hiddenDims[no - 1], outFactor * np.prod(dataShape), K=K, outFactor=outFactor, maskType="B", reshape=[outFactor, *dataShape[1:]]))
            else:
                layerList.append(MaskedMLPforSplineFlow(hiddenDims[no - 1], hiddenDims[no], K=K, outFactor=outFactor, maskType="B"))
            if activation[no] is not None:
                layerList.append(activation[no])

        self.layerList = nn.ModuleList(layerList)

    def forward(self, x):
        for layer in self.layerList:
            x = layer(x)
        return x
