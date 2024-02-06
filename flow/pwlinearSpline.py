import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .flow import Flow
from utils import unconstrained_linear_spline


class TpwLinearSpline(Flow):
    def __init__(self, networkList, maskList, K, B, split=None, fixFirst=None, prior=None, name="pwLinearSpline"):
        super(TpwLinearSpline, self).__init__(prior, name)
        self.K = K
        self.B = B
        self.maskList = nn.Parameter(maskList, requires_grad=False)
        self.networkList = nn.ModuleList(networkList)
        self.split = split

        assert len(self.maskList) == len(self.networkList)

        if fixFirst is not None:
            self.register_buffer('fixFirst', fixFirst)
        else:
            self.fixFirst = fixFirst

    def forward(self, x, T=None):
        logDet = torch.zeros(x.shape[0],1).to(x)
        dim = np.prod(x.shape[1:])
        shape = x.shape

        for i in range(len(self.networkList)):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            if self.split is None:
                lower = torch.masked_select(x, mask).reshape(*shape[:-1], int(mask.sum().item()//np.prod(x.shape[1:-1])))
                upper = torch.masked_select(x, maskR).reshape(*shape[:-1], int(maskR.sum().item()//np.prod(x.shape[1:-1])))
            else:
                lower = torch.masked_select(x, mask).reshape(*self.split[i][0])
                upper = torch.masked_select(x, maskR).reshape(*self.split[i][1])

            if self.fixFirst is not None and i % 2 == 0:
                assert torch.allclose(lower[:, :, 0, 0], self.fixFirst)

            if T is not None:
                if len(T.shape) >= 1:
                    _T = T.view(-1, *([1] * (len(upper.shape) - 1))).repeat(1, *(upper.shape[1:]))
                else:
                    _T = T.repeat(upper.shape[0], 1, *upper.shape[2:])
                upper = torch.cat([upper, _T], dim=1)

            q = self.networkList[i](upper)
            if self.split is None:
                q = q.reshape(shape[0], shape[1], self.K, shape[-2], int(mask.sum().item()//np.prod(x.shape[1:-1])))
            else:
                q = q.reshape(*self.split[i][0][:-2], self.K, *self.split[i][0][-2:])
            q = q.transpose(-3, -2).transpose(-2, -1) # put channel dim at the last

            if T is not None:
                if len(T.shape) >= 1:
                    T = T.reshape(T.shape[0], *([1] * (len(q.shape) - 1)))
                #q = q / T
            lower, ld = unconstrained_linear_spline(
                lower, unnormalized_pdf=q, inverse=False, tail_bound=self.B
            )

            if self.fixFirst is not None and i % 2 == 0:
                lower = torch.cat([x[:, :, 0, 0].unsqueeze(-1), lower.flatten(-2, -1)[:, :, 1:]], dim=-1).reshape(lower.shape)
                ld = ld.flatten(-2, -1)[:, :, 1:]

            logDet += ld.reshape(shape[0], -1).sum(1, keepdim=True)
            x = x.masked_scatter(mask, lower)

        return x, logDet

    def inverse(self, z, T=None):
        logDet = torch.zeros(z.shape[0], 1).to(z)
        dim = np.prod(z.shape[1:])
        shape = z.shape

        for i in reversed(range(len(self.networkList))):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            if self.split is None:
                lower = torch.masked_select(z, mask).reshape(*shape[:-1], int(mask.sum().item()//np.prod(z.shape[1:-1])))
                upper = torch.masked_select(z, maskR).reshape(*shape[:-1], int(maskR.sum().item()//np.prod(z.shape[1:-1])))
            else:
                lower = torch.masked_select(z, mask).reshape(*self.split[i][0])
                upper = torch.masked_select(z, maskR).reshape(*self.split[i][1])

            if self.fixFirst is not None and i % 2 == 0:
                assert torch.allclose(lower[:, :, 0, 0], self.fixFirst)

            if T is not None:
                if len(T.shape) >= 1:
                    _T = T.view(-1, *([1] * (len(upper.shape) - 1))).repeat(1, *(upper.shape[1:]))
                else:
                    _T = T.repeat(upper.shape[0], 1, *upper.shape[2:])
                upper = torch.cat([upper, _T], dim=1)

            q = self.networkList[i](upper)
            if self.split is None:
                q = q.reshape(shape[0], shape[1], self.K, shape[-2], int(mask.sum().item()//np.prod(z.shape[1:-1])))
            else:
                q = q.reshape(*self.split[i][0][:-2], self.K, *self.split[i][0][-2:])
            q = q.transpose(-3, -2).transpose(-2, -1) # put channel dim at the last
            if T is not None:
                if len(T.shape) >= 1:
                    T = T.reshape(T.shape[0], *([1] * (len(q.shape) - 1)))
                #q = q / T
            lower, ld = unconstrained_linear_spline(
                lower, unnormalized_pdf=q, inverse=True, tail_bound=self.B
            )

            if self.fixFirst is not None and i % 2 == 0:
                lower = torch.cat([z[:, :, 0, 0].unsqueeze(-1), lower.flatten(-2, -1)[:, :, 1:]], dim=-1).reshape(lower.shape)
                ld = ld.flatten(-2, -1)[:, :, 1:]

            logDet += ld.reshape(shape[0], -1).sum(1, keepdim=True)
            z = z.masked_scatter(mask, lower)

        return z, logDet


