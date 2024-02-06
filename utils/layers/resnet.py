import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResidualBlock(nn.Module):
    def __init__(self, channels, channelsOut=None, activation=F.relu):
        super(ConvResidualBlock, self).__init__()
        if channelsOut is None:
            channelsOut = channels
        self.activation = activation
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(channels, channelsOut, kernel_size=3, padding=1),
            nn.Conv2d(channelsOut, channelsOut, kernel_size=3, padding=1),
        ])

    def forward(self, inputs):
        temps = inputs
        temps = self.conv_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[1](temps)
        temps = inputs + temps
        temps = self.activation(temps)
        return temps


class ConvResidualNet(nn.Module):
    def __init__(self, inChannels, outChannels, hiddenChannels, numBlocks=2, activation=F.relu):
        super(ConvResidualNet, self).__init__()
        self.hiddenChannels = hiddenChannels
        self.initial = nn.Conv2d(
            in_channels=inChannels,
            out_channels=hiddenChannels,
            kernel_size=1,
            padding=0
        )
        self.blocks = nn.ModuleList([
            ConvResidualBlock(
                channels=hiddenChannels,
                activation=activation,
            ) for _ in range(numBlocks)
        ])
        self.final = nn.Conv2d(hiddenChannels, outChannels, kernel_size=1, padding=0)

    def forward(self, inputs):
        temps = self.initial(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final(temps)

        return outputs