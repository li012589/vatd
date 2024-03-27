import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import source, flow, utils
import h5py, os

from decoratedResnet import DecoratedResNet, SimpleMaskedMLPforSplineFlow

#torch.manual_seed(42)

# loading obej
load = None

autodiffLevel = 0 # 0 for rejAcptRep; 1 for priorReinforce; None direct repam

# parameters for XY dists
L = 16 # lattice length
T0 = 1. # base t for XY dist
J = 1. # interacting factor
factorLst = [0.5, 0.75, 0.85, 0.9, 1., 1.1, 1.25, 2.4] # multi factors for T0 to form temperatures points
Tlist = [term * T0 for term in factorLst]

# parameters for the beta range
betalow = 0.4
betahigh = 2.0

betaRange = betahigh - betalow

stepNum = 35

# parameters for cubic spline flow
K = 25 # the number of bins
pwLinearNnet = 16 # inner t net layer number
# for usage of MLP
pwLinearNetVector = [400, 500, 600] # t net hidden layer shape

# to fix the first spin
fixFirst = torch.tensor([[0.0]])

# for usage of resnet
kernelSize = 9
hiddenChannels = 128
hiddenConvLayers = 6
hiddenWidth = 128
hiddenFcLayers = 1

mappingBoundary = None
B = 1 # boundary parameter

if B != np.pi:
    mappingBoundary = np.pi / B

# parameters for optimization
lr = 7.e-4 # learning rate
eps = 1.e-8
lrdecay = 0.997
warmup = 500
batchSize = 1024 # training batch size
evalBatchSize = 2000 # evalutaion batch size
maxIter = 10000 # max step number
saveStep = 50 # save every n steps
clipGrad = 0.0
lamd1 = 5.0

# define computation device
#device = torch.device("cpu")
device = torch.device("cuda:0")

# create saving folder
name = "cubic_vonMises_Multi_" + str(len(factorLst)) + "_"+"XY_J"+str(J) + "_T"+str(T0)+"_L"+str(L)+"_t_"+str(factorLst).replace(', ','_').replace('[','').replace(']','')
rootFolder = os.path.join("opt", name)
utils.createWorkSpace(rootFolder)

# init XY at different temperatures
target = source.XYbreakSym(L=L, d=2, T=T0, J=J, flg2d=True, fixFirst=fixFirst.item()).to(device)

# prior parameters mapper
kappaMapper = utils.SoftplusWithBound(lowerBound=1e-3) # map kappa to lower bound if it's too low

if load is None:
    # init prior
    p1 = source.DeltaSource([1, 1], pos=fixFirst.view(-1))
    p2 = source.VonMisesImp([1, L**2 - 1], mu=0.1 * torch.randn([1, L * L - 1]), kappa=(torch.abs(torch.ones([1, L * L -1]) + 0.1 * torch.randn([1, L * L - 1]))), B=B, trainMu=True, trainKappa=True, kappaMapper=kappaMapper)
    prior = source.ConcatenatedSource([p1, p2], dim=-1, reshape=[1, L, L], directLogProb=True)

    # init cubice spline
    maskList = []
    netList = []
    # checkboard
    for n in range(pwLinearNnet // 3):
        if n % 2 == 0:
            b = torch.zeros(1, L//2, L//2, 2, 2)
            b[:, :, :, 0, 0] = 1
            b[:, :, :, 1, 1] = 1
            b = b.permute([0, 1, 3, 2, 4]).reshape(1, L, L)
            netList.append(DecoratedResNet(L, 2, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU()))
        else:
            b = 1 - b
            netList.append(DecoratedResNet(L, 2, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU()))
        maskList.append(b)

    maskList = torch.cat(maskList, 0).to(torch.uint8)
    tpwLinear = flow.CubicSpline(netList, maskList, K=K, B=B, fixFirst=fixFirst)

    joint = flow.CombinedFlow([tpwLinear], prior=prior, mappingBoundary=mappingBoundary).to(device)

    bestTrainLoss = 99999999
else:
    joint = torch.load(load, map_location=device)
    print("Model loaded at:", load)

    # evaluation loaded model
    lossLst = []
    for t in Tlist:
        with torch.no_grad():
            T = torch.tensor(t).to(device)
            samples, logProb = joint.sample(evalBatchSize, T=T)
            loss = (logProb + target.energyWithT(samples, T=T)).mean(dim=0, keepdim=True)
        lossLst.append(loss.detach().item())

    lossLst = np.array(lossLst)
    lossSum = lossLst.sum()

    bestTrainLoss = lossSum

    # feeddback
    printString = "Loaded model, L: %.5f, "
    resultLst = [lossSum]
    for idx, factor in enumerate(factorLst):
        printString += "XY@" + str(factor) + ": %.2f, "
        resultLst += [lossLst[idx].item()]
    resultLst = tuple(resultLst)
    print(printString % resultLst)

params = list(joint.parameters())
params = list(filter(lambda p: p.requires_grad, params))
nparams = sum([np.prod(p.size()) for p in params])
print('total nubmer of trainable parameters:', nparams)

# init optimizer
optimizer = torch.optim.Adamax(params, lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.8)

# start optimize
LOSS = []
for e in range(maxIter):
    try:
        tstart = time.time()
        # training
        for s in range(stepNum):
            optimizer.zero_grad()
            beta = (torch.rand(batchSize) * betaRange + betalow).to(device)

            z = joint.prior.sample(batchSize, T=1/beta)
            if autodiffLevel == 1:
                z = z.detach()
            zlogProb = joint.prior.logProbability(z, T=1/beta)
            samples, logDet = joint.forward(z, T=1/beta)
            logProb = zlogProb - logDet

            loss = (logProb + target.energyWithT(samples, T=1/beta))

            if autodiffLevel == 0:
                gCorr = (loss.detach() - loss.detach().mean()) * joint.prior.sourceList[1].logAceptRej()
            elif autodiffLevel == 1:
                gCorr = (loss.detach() - loss.detach().mean()) * zlogProb
            elif autodiffLevel is None:
                gCorr = 0
            lossReinforce = loss + gCorr
            lossReinforce = lossReinforce.mean(0, keepdim=True)

            lossReinforce.backward()

            if clipGrad != 0:
                nn.utils.clip_grad_norm_(params, clipGrad)

            optimizer.step()
            # ramp up the kappa if it gets too small to get gradients
            joint.prior.sourceList[1].kappa.data[torch.where(joint.prior.sourceList[1].kappa<=4e-4)] = 4.5e-4

        trainTime = time.time() - tstart

        # evaluation
        lossLst = []
        for t in Tlist:
            with torch.no_grad():
                T = torch.tensor(t).to(device)
                samples, logProb = joint.sample(evalBatchSize, T=T)
                loss = (logProb + target.energyWithT(samples, T=T)).mean(dim=0, keepdim=True)
            lossLst.append(loss.detach().item())

    except Exception as err:
        torch.save(joint, os.path.join(rootFolder, 'crush_saved.saving'))
        print(err)
        import pdb
        pdb.set_trace()

    lossLst = np.array(lossLst)
    lossSum = lossLst.sum()

    LOSS.append(lossSum)
    if lossSum < bestTrainLoss:
        bestTrainLoss = lossSum.item()
        torch.save(joint, os.path.join(rootFolder, 'best_TrainLoss_joint.saving'))
        print("--> Updated best model: ", bestTrainLoss)

    # feeddback
    printString = "epoch: %d, L: %.5f, "
    resultLst = [e, lossSum]
    for idx, factor in enumerate(factorLst):
        printString += "XY@" + str(factor) + ": %.2f,"
        resultLst += [lossLst[idx].item()]
    printString += "time: %.2f, best: %.5f"
    resultLst += [trainTime, bestTrainLoss]
    resultLst = tuple(resultLst)
    print(printString % resultLst)

    # step the schedular
    scheduler.step()

    # save
    if e % saveStep == 0 or e == 0:
        # save joint and opt
        torch.save(joint, os.path.join(rootFolder, 'savings', joint.name + "_epoch_" + str(e) + ".saving"))
        torch.save(optimizer, os.path.join(rootFolder, 'savings', joint.name + "_epoch_" + str(e) + "_opt.saving"))
        # save loss values
        with h5py.File(os.path.join(rootFolder, "records", "LOSS"+'.hdf5'), 'w') as f:
            f.create_dataset("LOSS", data=np.array(LOSS))
        # plot loss curve
        lossfig = plt.figure(figsize=(8, 5))
        lossax = lossfig.add_subplot(111)
        epoch = len(LOSS)
        lossax.plot(np.arange(epoch), np.array(LOSS), 'go-', label="loss", markersize=2.5)
        lossax.set_xlim(0, epoch)
        lossax.legend()
        lossax.set_title("Loss Curve")
        plt.savefig(os.path.join(rootFolder, 'pic', 'lossCurve.png'), bbox_inches="tight", pad_inches=0)
        plt.close()
        # clean extra saving
        utils.cleanSaving(rootFolder, e, 6 * saveStep, joint.name)
