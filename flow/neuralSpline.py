import torch
from torch import  nn
from torch.nn import functional as F
import numpy as np
from .flow import Flow
from utils import unconstrained_RQS


class NeuralSpline(Flow):
    def __init__(self, networkList, maskList, K = 5, B = 3, fixFirst=None, prior=None, name="NeuralSpline"):
        super(NeuralSpline, self).__init__(prior, name)
        self.K = K
        self.B = B
        self.maskList = nn.Parameter(maskList, requires_grad=False)
        self.networkList = nn.ModuleList(networkList)

        if fixFirst is not None:
            self.register_buffer('fixFirst', fixFirst)
        else:
            self.fixFirst = fixFirst

        assert len(self.maskList) == len(self.networkList)

    def forward(self, x, T=None, dilute=2):
        logDet = torch.zeros(x.shape[0],1).to(x)
        dim = np.prod(x.shape[1:])
        shape = x.shape

        for i in range(len(self.networkList)):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            lower = torch.masked_select(x, mask).reshape(*shape[:-1], int(mask.sum().item()//np.prod(x.shape[1:-1])))
            upper = torch.masked_select(x, maskR).reshape(*shape[:-1], int(maskR.sum().item()//np.prod(x.shape[1:-1])))

            if self.fixFirst is not None and i % 2 == 0:
                assert torch.allclose(lower[:, :, 0, 0], self.fixFirst)

            if T is not None:
                _T = torch.zeros(*upper.shape[:2], upper.shape[-2]//dilute, upper.shape[-1]//dilute, dilute, dilute).to(T)
                if len(T.shape) < 1:
                    T = T.repeat(upper.shape[0])
                _T[:, :, :, :, 0, 0] = T.view([-1] + [1] * (len(upper.shape) - 1)).repeat(1, *_T.shape[1:4])
                _T = _T.permute(0, 1, 2, 4, 3, 5).reshape(*upper.shape)
                upper = torch.cat([upper, _T], dim=1)

            out = self.networkList[i](upper).reshape(lower.shape[0], lower.shape[1], 3 * self.K - 1, *lower.shape[-2:]).permute([0, 1, 3, 4, 2])

            W, H, D = torch.split(out, self.K, dim = -1)

            lower, ld = unconstrained_RQS(
                lower, W, H, D, inverse=False, tail_bound=self.B)

            if self.fixFirst is not None and i % 2 == 0:
                lower = torch.cat([x[:, :, 0, 0].unsqueeze(-1), lower.flatten(-2, -1)[:, :, 1:]], dim=-1).reshape(lower.shape)
                ld = ld.flatten(-2, -1)[:, :, 1:]

            logDet += ld.reshape(shape[0],-1).sum(1, keepdim=True)
            x = x.masked_scatter(mask, lower)

        return x, logDet

    def inverse(self, z, T=None, dilute=2):
        logDet = torch.zeros(z.shape[0], 1).to(z)
        dim = np.prod(z.shape[1:])
        shape = z.shape

        for i in reversed(range(len(self.networkList))):
            maskR = (1 - self.maskList[i]).bool()
            mask = self.maskList[i].bool()
            lower = torch.masked_select(z, mask).reshape(*shape[:-1], int(mask.sum().item()//np.prod(z.shape[1:-1])))
            upper = torch.masked_select(z, maskR).reshape(*shape[:-1], int(maskR.sum().item()//np.prod(z.shape[1:-1])))

            if self.fixFirst is not None and i % 2 == 0:
                assert torch.allclose(lower[:, :, 0, 0], self.fixFirst)

            if T is not None:
                _T = torch.zeros(*upper.shape[:2], upper.shape[-2]//dilute, upper.shape[-1]//dilute, dilute, dilute).to(T)
                if len(T.shape) < 1:
                    T = T.repeat(upper.shape[0])
                _T[:, :, :, :, 0, 0] = T.view([-1] + [1] * (len(upper.shape) - 1)).repeat(1, *_T.shape[1:4])
                _T = _T.permute(0, 1, 2, 4, 3, 5).reshape(*upper.shape)
                upper = torch.cat([upper, _T], dim=1)

            out = self.networkList[i](upper).reshape(lower.shape[0], lower.shape[1], 3 * self.K - 1, *lower.shape[-2:]).permute([0, 1, 3, 4, 2])

            W, H, D = torch.split(out, self.K, dim = -1)

            lower, ld = unconstrained_RQS(
                lower, W, H, D, inverse=True, tail_bound=self.B)

            if self.fixFirst is not None and i % 2 == 0:
                lower = torch.cat([z[:, :, 0, 0].unsqueeze(-1), lower.flatten(-2, -1)[:, :, 1:]], dim=-1).reshape(lower.shape)
                ld = ld.flatten(-2, -1)[:, :, 1:]

            logDet += ld.reshape(shape[0],-1).sum(1, keepdim=True)
            z = z.masked_scatter(mask, lower)

        return z, logDet

