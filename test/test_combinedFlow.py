from flowRelated import *

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np
import utils
import flow
import source
import utils
from copy import deepcopy


def test_bijective():
    p = source.Gaussian([3, 32, 32])
    T = torch.tensor(5.0)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f1 = flow.NICE(maskList, tList)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f2 = flow.NICE(maskList, tList)

    flowList = [f1, f2]

    f = flow.CombinedFlow(flowList, prior=p)

    bijective(f, decimal=3)
    bijective(f, decimal=3, args=[T])

def test_detJacobian():
    p = source.Gaussian([3, 8, 8])
    T = torch.tensor(5.0)

    maskList = []
    for n in range(2):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 8 * 4), torch.ones(3 * 8 * 4)])[torch.randperm(3 * 8 * 8)].reshape(1, 3, 8, 8)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 8 * 4, 20, 50, 3 * 8 * 4], [nn.ELU(), nn.ELU(), None]) for _ in range(2)]
    f1 = flow.NICE(maskList, tList)

    maskList = []
    for n in range(2):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 8 * 4), torch.ones(3 * 8 * 4)])[torch.randperm(3 * 8 * 8)].reshape(1, 3, 8, 8)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 8 * 4, 20, 50, 3 * 8 * 4], [nn.ELU(), nn.ELU(), None]) for _ in range(2)]
    f2 = flow.NICE(maskList, tList)

    flowList = [f1, f2]

    f = flow.CombinedFlow(flowList, prior=p)

    detJacobian(f)
    detJacobian(f, args=[T])


def test_saveload():
    p = source.Gaussian([3, 32, 32])

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f1 = flow.NICE(maskList, tList)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f2 = flow.NICE(maskList, tList)

    flowList = [f1, f2]

    f = flow.CombinedFlow(flowList, prior=p)

    p = source.Gaussian([3, 32, 32])

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f1 = flow.NICE(maskList, tList)

    maskList = []
    for n in range(4):
        if n % 2 == 0:
            b = torch.cat([torch.zeros(3 * 32 * 16), torch.ones(3 * 32 * 16)])[torch.randperm(3 * 32 * 32)].reshape(1, 3, 32, 32)
        else:
            b = 1 - b
        maskList.append(b)
    maskList = torch.cat(maskList, 0).to(torch.float32)
    tList = [utils.SimpleMLPreshape([3 * 32 * 16, 200, 500, 3 * 32 * 16], [nn.ELU(), nn.ELU(), None]) for _ in range(4)]
    f2 = flow.NICE(maskList, tList)

    flowList = [f1, f2]

    blankf = flow.CombinedFlow(flowList, prior=p)

    saveload(f,blankf, decimal=3)

if __name__ == "__main__":
    #test_bijective()
    #test_detJacobian()
    test_saveload()
