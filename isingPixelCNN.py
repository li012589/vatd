import time
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from source import VanillaIsing
from autoregressive import DiscretePixelCNN
from autoregressive import MaskedResConv
from utils import isingLogz, isingLogzTr
import source, utils
import h5py, os, sys

## subfunctions
def mapping(batch):
    return batch * 2 - 1

def reverseMapping(batch):
    return torch.div(batch + 1, 2, rounding_mode='trunc')

fixFirst = 1
# parameters for dists
L = 16 # Lattice length
T0 = 2.269 # scaling T0
factorLst = [0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.8] # multiple factors for T0 to form factor points
TList = [term * T0 for term in factorLst]
TempStart, TempEnd, TempStep = 0.8, 10.8, 10
TRange = torch.arange(TempStart, TempEnd, TempStep)
betaRange = 1 / TRange
dataSize = betaRange.shape[0]
betaBatchSize = 1
stepNum = dataSize // betaBatchSize

# params for optimization
lr = 1.e-3
eps = 1.e-8
lrdecay = 1000
batchSize = 1
maxEpoch = 4600
saveStep = 1
clipGrad = 1.0 #clip gradient to stablize, 0 for not using

# define device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# create saving folder
name = 'isingTraining'
rootFolder = './opt/'+ name + '/'
utils.createWorkSpace(rootFolder)

# params for Discrete Pixel CNN
channel = 1
kernelSize = 9
hiddenChannels = 64
hiddenConvLayers = 6
hiddenKernelSize = 9
hiddenWidth = 64
hiddenFcLayers = 2
category = 2
augmentChannels = 1

maskedConv = MaskedResConv(channel, kernelSize, hiddenChannels, hiddenConvLayers, hiddenKernelSize, hiddenWidth, hiddenFcLayers, category, augmentChannels=augmentChannels).to(device)

target = source.VanillaIsing(L=L, d=2, T=T0, breakSym=True).to(device)
isingExactloss = [isingLogzTr(n = L, j=1.0, beta = (1/torch.tensor(T))).item() for T in TList]

model = DiscretePixelCNN([L, L], channel, maskedConv, fixFirst=fixFirst, mapping=mapping, reverseMapping=reverseMapping, device=device, name=name)

params = list(model.parameters())
params = list(filter(lambda p: p.requires_grad, params))
nparams = sum([np.prod(p.size()) for p in params])
print('total nubmer of trainable parameters:', nparams)

# init optimizer
optimizer = torch.optim.Adam(params, lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.6)

# Training
LOSS = []
bestTrainLoss = 99999999

for e in range(maxEpoch):

    tstart = time.time()
    for s in range(stepNum):
        idxLst = torch.randint(0, dataSize, (betaBatchSize, ))
        optimizer.zero_grad()
        for idx in idxLst:
            with torch.no_grad():
                samples = model.sample(batchSize=batchSize, T=1/betaRange[idx])
            logProb = model.logProbability(samples, T=1/betaRange[idx])
            with torch.no_grad():
                loss = (logProb + target.energyWithT(samples, T=1/betaRange[idx]))
            lossReinforce = torch.mean((loss - loss.mean()) * logProb)
            lossReinforce.backward()

        if clipGrad !=0:
            nn.utils.clip_grad_norm_(params, clipGrad)

        optimizer.step()
    trainTime =  time.time() - tstart

    # evaluation
    lossLst = []
    for T in TList:
        T = torch.tensor(T).to(device)
        with torch.no_grad():
            samples = model.sample(batchSize, T=T)
            logProb = model.logProbability(samples, T=T)
            loss = (logProb + target.energyWithT(samples, T=T))
        lossLst.append(loss.mean().detach().item())
    scheduler.step()
    lossLst = np.array(lossLst)
    lossSum = lossLst.sum()

    # opt steps
    LOSS.append(lossSum)
    if lossSum < bestTrainLoss:
        bestTrainLoss = lossSum
        torch.save(model, rootFolder + 'best_TrainLoss_model.saving')

    # feedback
    printString = 'epoch: {:d}, L: {:.5f},'
    resultLst = [e, lossSum]
    for idx, factor in enumerate(factorLst):
        printString += 'ising@' +  str(factor) + ':{:.5f},' + 'err:{:.2f},'
        resultLst +=[lossLst[idx].item()/16/16/0.45, lossLst[idx].item() + isingExactloss[idx]]

    printString +='time:{:.2f}'
    resultLst += [trainTime]
    resultLst = tuple(resultLst)
    print(printString.format(*resultLst))

    # save
    if e % saveStep == 0 or e == 0:
        # save model and opt
        torch.save({
            'Epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, rootFolder + 'savings/' + model.name + "_epoch_" + str(e) + ".saving")

        # save loss values
        with h5py.File(rootFolder + 'records/' + 'LOSS'  + '.hdf5', 'w') as f:
            f.create_dataset('LOSS', data=np.array(LOSS))

        # plot loss curve
        lossfig = plt.figure(figsize=(12,5))
        lossax = lossfig.add_subplot(111)
        epoch = len(LOSS)
        lossax.plot(np.arange(epoch), np.array(LOSS), 'go-', label="loss", markersize=2.5)
        lossax.set_xlim(0, epoch)
        lossax.tick_params(axis='both', labelsize=11)
        lossax.set_xlabel('epoch',  fontsize=13)
        lossax.set_ylabel('Loss',  fontsize=13)
        lossax.legend(fontsize=14)
        lossax.set_title('LOSS Curve', fontsize=15)
        plt.savefig(rootFolder + 'pic/lossCurve.png', bbox_inches="tight", pad_inches=0)
        plt.close()
        # clean extra saving
        utils.cleanSaving(rootFolder, e, 6 * saveStep, model.name)