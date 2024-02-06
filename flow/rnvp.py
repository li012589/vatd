import torch
from torch import  nn
import numpy as np
from numpy.testing import  assert_array_equal
from .flow import Flow

class RNVP(Flow):
    def __init__(self, maskList, tList, sList, prior=None, name="RNVP"):
        super(RNVP, self).__init__(prior, name)

        assert len(tList) == len(sList)
        assert len(tList) == len(maskList)

        self.maskList = nn.Parameter(maskList, requires_grad=False)

        self.tList = torch.nn.ModuleList(tList)
        self.sList = torch.nn.ModuleList(sList)

    def forward(self,z):
        inverseLogjac = z.new_zeros(z.shape[0], 1)

        for i in range(len(self.tList)):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            zA = torch.masked_select(z, mask).reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3] // 2)
            zB = torch.masked_select(z, maskR).reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3] // 2)

            s = self.sList[i](zB)
            t = self.tList[i](zB)
            assert_array_equal(t.shape, zA.shape)
            assert_array_equal(s.shape, zA.shape)

            zA = zA * torch.exp(s) + t
            z = z.masked_scatter(mask, zA).contiguous()

            for k in range(z.shape[0]):
                inverseLogjac[k, 0] += s[k].sum()
        return z, inverseLogjac

    def inverse(self,y):
        forwardLogjac = y.new_zeros(y.shape[0], 1)

        for i in reversed(range(len(self.tList))):
            maskR = (1-self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            yA = torch.masked_select(y, mask).reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3] // 2)
            yB = torch.masked_select(y, maskR).reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3] // 2)

            s=self.sList[i](yB)
            t=self.tList[i](yB)

            assert_array_equal(t.shape, yB.shape)
            assert_array_equal(s.shape, yA.shape)

            yA = (yA - t) * torch.exp(-s)
            y=y.masked_scatter(mask, yA).contiguous()

            for k in range(y.shape[0]):
                forwardLogjac[k, 0] -= s[k].sum()
        return y, forwardLogjac


