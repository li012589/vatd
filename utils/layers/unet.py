import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ConvResidualBlock


def doubleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, inChannels, intermediaChannels=None, outChannels=None, maxHiddenChannels=512, numBlocks=2, useRes=True):
        super(UNet, self).__init__()
        maxNumBlocks = np.log2(maxHiddenChannels)
        assert maxNumBlocks % 1 == 0 and maxNumBlocks > numBlocks

        self.numBlocks = numBlocks

        if intermediaChannels is not None:
            if useRes:
                self.inital = ConvResidualBlock(inChannels, intermediaChannels)
            else:
                self.inital = doubleConv(inChannels, intermediaChannels)
            inChannels = intermediaChannels
        else:
            self.inital = None

        if useRes:
            downConvLst = [ConvResidualBlock(inChannels, int(2**(maxNumBlocks - numBlocks)))]
            self.downConv = nn.ModuleList(
                downConvLst
                +
                [ConvResidualBlock(
                    channels = int(2**(maxNumBlocks - numBlocks + i)),
                    channelsOut = int(2**(maxNumBlocks - numBlocks + 1 + i))
                )
                for i in range(numBlocks)
            ])
        else:
            downConvLst = [doubleConv(inChannels, int(2**(maxNumBlocks - numBlocks)))]
            self.downConv = nn.ModuleList(
                downConvLst
                +
                [doubleConv(
                    in_channels = int(2**(maxNumBlocks - numBlocks + i)),
                    out_channels = int(2**(maxNumBlocks - numBlocks + 1 + i))
                )
                for i in range(numBlocks)
            ])

        self.conv1by1 = nn.ModuleList(
            [nn.Conv2d(int(2**(maxNumBlocks - numBlocks)),
                       int(2**(maxNumBlocks - numBlocks)), 1, padding=0),
            ] + [nn.Sequential(
                nn.Conv2d(int(2**(maxNumBlocks - numBlocks + i + 1)),
                          int(2**(maxNumBlocks - numBlocks + i + 1)), 1, padding=0),
                nn.ReLU(inplace=True))
            for i in range(numBlocks)
            ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpooling = nn.MaxPool2d(2)

        if useRes:
            self.upConv = nn.ModuleList(
                [ConvResidualBlock(
                    channels = int(2**(maxNumBlocks - numBlocks + i) + 2**(maxNumBlocks - numBlocks + i + 1)),
                    channelsOut = int(2**(maxNumBlocks - numBlocks + i))
                )
                for i in range(numBlocks -1, -1, -1)
                ] + [ConvResidualBlock(
                    channels = int(2**(maxNumBlocks - numBlocks)) + inChannels,
                    channelsOut = inChannels)]
            )
        else:
            self.upConv = nn.ModuleList(
                [doubleConv(
                    in_channels = int(2**(maxNumBlocks - numBlocks + i) + 2**(maxNumBlocks - numBlocks + i + 1)),
                    out_channels = int(2**(maxNumBlocks - numBlocks + i))
                )
                for i in range(numBlocks -1, -1, -1)
                ] + [doubleConv(
                    in_channels = int(2**(maxNumBlocks - numBlocks)) + inChannels,
                    out_channels = inChannels)]
            )

        if outChannels is not None:
            if useRes:
                self.final = nn.Sequential(
                    ConvResidualBlock(inChannels, outChannels),
                    nn.Conv2d(outChannels, outChannels, 1)
                )
            else:
                self.final = nn.Sequential(
                    doubleConv(inChannels, outChannels),
                    nn.Conv2d(outChannels, outChannels, 1)
                )
        else:
            self.final = None


    def forward(self, inputs):

        if self.inital is not None:
            inputs = self.inital(inputs)

        context = [inputs]
        for i, down in enumerate(self.downConv):
            inputs = down(inputs)
            inputs = self.maxpooling(inputs)
            inputs = self.conv1by1[i](inputs)
            context.append(inputs)

        for i, term in enumerate(reversed(context[:-1])):
            inputs = self.upsample(inputs)
            inputs = torch.cat([inputs, term], dim=1)
            inputs = self.upConv[i](inputs)

        if self.final is not None:
            inputs = self.final(inputs)

        return inputs

