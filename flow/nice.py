import torch
from torch import nn
from numpy.testing import assert_array_equal
from .flow import Flow

'''
obsolete! use TRNVP instead
'''
class NICE(Flow):
    def __init__(self, maskList, tList, B=None, prior = None, name = "NICE"):
        super(NICE,self).__init__(prior, name)

        assert len(tList) == len(maskList)

        self.maskList = nn.Parameter(maskList,requires_grad=False)
        self.B = B

        self.tList = torch.nn.ModuleList(tList)

    def forward(self, z, T=None):
        inverseLogjac = z.new_zeros(z.shape[0], 1)
        for i in range(len(self.tList)):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            zA = torch.masked_select(z, mask).reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3] // 2)
            zB = torch.masked_select(z, maskR).reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3] // 2)

            t = (self.tList[i](zB))
            assert_array_equal(t.shape, zA.shape)

            zA = zA + t
            if self.B is not None:
                zA = (zA + self.B) % (2 * self.B) - self.B
            z = z.masked_scatter(mask, zA).contiguous()
        return z, inverseLogjac

    def inverse(self, y, T=None):
        forwardLogjac = y.new_zeros(y.shape[0], 1)
        for i in reversed(range(len(self.tList))):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            yA = torch.masked_select(y, mask).reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3] // 2)
            yB = torch.masked_select(y, maskR).reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3] // 2)

            t = (self.tList[i](yB))
            assert_array_equal(t.shape, yA.shape)

            yA = yA - t
            if self.B is not None:
                yA = (yA + self.B) % (2 * self.B) - self.B

            y = y.masked_scatter(mask, yA).contiguous()
        return y, forwardLogjac
