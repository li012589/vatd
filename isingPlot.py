import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from source import VanillaIsing
from autoregressive import DiscretePixelCNN
from autoregressive import MaskedResConv
from utils import isingLogz, isingLogzTr
import source, utils, os, sys

def exactCal(L, beta):
    beta = beta.clone().detach().requires_grad_(True)
    logZexact = utils.isingLogzTr(L, 1.0, beta)
    return logZexact.detach().item()


def reinforce(model, beta, L, batchSize, device):
    beta = beta.clone().detach().repeat(batchSize, 1).requires_grad_(True).to(device)
    T = 1/beta
    ising = source.VanillaIsing(L=L, d=2, T=T, exact=False).to(device)
    with torch.no_grad():
        samples = model.sample(batchSize, T=T)
        energyWithT = ising.energyWithT(samples, T=T)
        energy = ising.energy(samples)
        magMean = (samples.reshape(batchSize, -1).sum(-1)).mean()/L**2
        mag2Mean = ((samples.reshape(batchSize, -1).sum(-1))**2).mean()/L**4

    logProb = model.logProbability(samples, T=T)
    with torch.no_grad():
        loss = logProb + energyWithT
        logZ = -(loss).mean().detach().item()
        partFnErr = loss.std().item()

    dlogProb = torch.autograd.grad(logProb, beta, grad_outputs=torch.ones(
        batchSize, 1).to(device), create_graph=True, allow_unused=True, retain_graph=True)[0]

    d2logProb = torch.autograd.grad(dlogProb, beta, grad_outputs=torch.ones(
        batchSize, 1).to(device), create_graph=True, allow_unused=True)[0]
    with torch.no_grad():
        meanEnergy = torch.mean((logProb + energyWithT + logZ)
                            * dlogProb + energy).detach().item()

        heatest1der = torch.mean(logProb*dlogProb).detach().item()*beta.detach().mean().item()

        term1 = torch.mean((logProb + energyWithT + logZ)*(d2logProb + dlogProb**2)).detach().item()

        term2 = torch.mean(dlogProb**2).detach().item()

        term3 = torch.mean((energy - energy.mean()) * dlogProb).detach().item()

        heatCap = - beta.detach().mean().item()**2 * (term1 + term2 + 2*term3)

    with torch.no_grad():
        b1 = torch.mean((logProb + energyWithT)*dlogProb**2)/torch.mean(dlogProb**2)
        b2 = torch.mean((logProb + energyWithT)*(d2logProb + dlogProb**2)**2)/torch.mean((d2logProb + dlogProb**2)**2)
        b3 = torch.mean((energy)*dlogProb**2)/torch.mean(dlogProb**2)
        meanEnergy_b = torch.mean((logProb + energyWithT -b1)
                            * dlogProb + energy).detach().item()
        term1_b = torch.mean((logProb + energyWithT -b2)*(d2logProb + dlogProb**2)).detach().item()
        term2_b = torch.mean(dlogProb**2).detach().item()
        term3_b = torch.mean((energy - b3) * dlogProb).detach().item()
        heatCap_b = - beta.detach().mean().item()**2 * (term1_b + term2_b + 2*term3_b)
    return logZ, meanEnergy, heatCap, heatest1der, magMean, mag2Mean, partFnErr, heatest1der, meanEnergy_b, heatCap_b, heatest1der


def heatCapPlot(model, L, batchSize, betaMin=0.3, betaMax=0.6, stepNum=100, savePath=None, show=False, device=torch.device('cpu'), autodiff=True, loop=1):
    betaLst = torch.linspace(betaMin, betaMax, steps=stepNum)
    if savePath is None:
        savePath = "./"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    if autodiff is True:
        savePath = savePath + 'autodiff_'
    else:
        savePath = savePath + 'stat_'

    exactlnZlst = []

    lnZlst = []
    heatClst = []
    meanElst = []
    heatCest1orderLst = []
    meanEBlst = []
    heatCBlst = []
    magLst = []
    mag2Lst = []
    partFnErrLst = []

    for beta in betaLst:

        beta = beta.clone().detach()
        logZexact = exactCal(L, beta)
        exactlnZlst.append(logZexact/L**2)

        if autodiff:
            lnZLoopLst = []
            meanELoopLst = []
            CvLoopLst = []
            magLoopLst = []
            mag2LoopLst = []
            meanELoopBlst = []
            CvLoopBlts = []
            heatCest1orderLoopLst = []
            partFnErrLoopLst = []
            for l in range(loop):
                logZ, meanEnergy, heatCap, heatest1der, magMean, mag2Mean, partFnErr, heatest1der, meanEnergy_b, heatCap_b, heatest1der = reinforce(model=model, beta=beta, L=L, batchSize=batchSize, device=device)
                lnZLoopLst.append(logZ)
                meanELoopLst.append(meanEnergy)
                CvLoopLst.append(heatCap)
                magLoopLst.append(magMean.item())
                mag2LoopLst.append(mag2Mean.item())
                meanELoopBlst.append(meanEnergy_b)
                CvLoopBlts.append(heatCap_b)
                heatCest1orderLoopLst.append(heatest1der)
                partFnErrLoopLst.append(partFnErr)

            lnZlst.append(np.array(lnZLoopLst).mean()/L**2)
            meanElst.append(np.array(meanELoopLst).mean()/L**2)
            heatClst.append(np.array(CvLoopLst).mean()/L**2)

            magLst.append(np.array(magLoopLst).mean())
            mag2Lst.append(np.array(mag2LoopLst).mean())
            meanEBlst.append(np.array(meanELoopBlst).mean()/L**2)
            heatCBlst.append(np.array(CvLoopBlts).mean()/L**2)
            heatCest1orderLst.append(np.array(heatCest1orderLoopLst).mean()/L**2)
            partFnErrLst.append(np.array(partFnErrLoopLst).mean()/L**2)
        else:
            eLoopLst = []
            e2LoopLst = []
            lnZLoopLst = []
            for l in range(loop):
                beta.repeat(batchSize, 1).requires_grad_(True).to(device)
                T = 1/beta
                ising = source.VanillaIsing(L=L, d=2, T=T, exact=False).to(device)
                with torch.no_grad():
                    samples = model.sample(batchSize, T=T)
                    logProb = model.logProbability(samples, T=T)
                    energy = ising.energy(samples)
                    logZ = -(logProb + energy * beta).mean().detach().item()
                eLoopLst.append(energy.mean().item())
                e2LoopLst.append((energy**2).mean().item())
                lnZLoopLst.append(logZ)

            meanE = np.array(eLoopLst).mean()/L**2
            Cv = (np.array(e2LoopLst).mean() - np.array(eLoopLst).mean()**2)*beta.detach().mean().item()**2/L**2
            logZLoop = np.array(lnZLoopLst).mean()


            lnZlst.append(logZLoop/L**2)
            meanElst.append(meanE)
            heatClst.append(Cv)

        error = abs(logZexact - logZ)
        if autodiff:
            print("beta:", beta.mean().item(), "meanE:",
                meanElst[-1], "Cv:", heatClst[-1], "error:", error, 'mag', magLst[-1], 'mag2', mag2Lst[-1], 'meanE_b:', meanEBlst[-1], 'Cv_b:', heatCBlst[-1], '1stEstCv:', heatCest1orderLst[-1], 'std_lnZ:', partFnErrLst[-1])
        else:
            print("beta:", beta.mean().item(), "meanE:",
                meanElst[-1], "Cv:", heatClst[-1], "error:", error)

    lnZlst = np.array(lnZlst)
    meanElst = np.array(meanElst)
    heatClst = np.array(heatClst)

    exactlnZlst = np.array(exactlnZlst)

    betaLst = betaLst.cpu().numpy()

    with open(savePath + "isingplotdata.npy", "wb") as f:
        np.save(f, betaLst)
        np.save(f, lnZlst)
        np.save(f, meanElst)
        np.save(f, heatClst)
        if autodiff:
            np.save(f, np.array(heatCest1orderLst))
            np.save(f, np.array(magLst))
            np.save(f, np.array(mag2Lst))
            np.save(f, np.array(meanEBlst))
            np.save(f, np.array(heatCBlst))
            np.save(f, np.array(heatCest1orderLst))
            np.save(f, np.array(partFnErrLoopLst))

    plt.figure(figsize=(12, 9))
    plt.title('lnZ')
    plt.xlabel('beta')
    plt.ylabel('lnZ')
    plt.scatter(betaLst, lnZlst, marker='o', label="est.")
    plt.scatter(betaLst, exactlnZlst, marker='+', label="exact")
    plt.legend(loc='best')
    plt.savefig(savePath + 'isinglnZ.pdf')

    plt.figure(figsize=(12, 9))
    plt.title('Mean Energy')
    plt.xlabel('beta')
    plt.ylabel('Mean Energy')
    plt.scatter(betaLst, meanElst, marker='o', label="est.")
    plt.legend(loc='best')
    plt.savefig(savePath + 'isingmeanE.pdf')

    plt.figure(figsize=(12, 9))
    plt.title('Heat Capacity')
    plt.xlabel('beta')
    plt.ylabel('Heat Capacity')
    plt.scatter(betaLst, heatClst, marker='o', label="est.")
    plt.legend(loc='best')
    plt.savefig(savePath + 'isingheatC.pdf')

    if show:
        plt.show()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parameters for dists
    L = 16  # lattice length
    num = 30  # plot points number
    batchSize = 2000 # the batch size to estimate free energy
    betaMin = 0.1
    betaMax = 1.0
    loop = 10

    # subfunctions
    def mapping(batch):
        return batch * 2 - 1

    def reverseMapping(batch):
        return torch.div(batch + 1, 2, rounding_mode='trunc')

    loadPath = 'opt/isingTraining/'
    model = torch.load(loadPath+"best_TrainLoss_model.saving", map_location=device).to(device)

    # evaluation
    T0 =2.269 # scaling T0
    factorLst = [0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.8] # multiple factors for T0 to form factor points
    TList = [term * T0 for term in factorLst]
    target = source.VanillaIsing(L=L, d=2, T=T0, breakSym=True).to(device)
    isingExactloss = [isingLogzTr(n=L, j=1.0, beta=(1/torch.tensor(T))).item() for T in TList]
    lossLst = []
    for T in TList:
        T = torch.tensor(T).to(device)
        with torch.no_grad():
            samples = model.sample(batchSize, T=T)
            logProb = model.logProbability(samples, T=T)
            loss = (logProb + target.energyWithT(samples, T=T))
        lossLst.append(loss.mean().detach().item())
    lossLst = np.array(lossLst)
    lossSum = lossLst.sum()

    printString = 'L: {:.5f},'
    resultLst = [lossSum]
    for idx, factor in enumerate(factorLst):
        printString += 'ising@' +  str(factor) + ':{:.5f},' + 'err:{:.2f},'
        resultLst += [lossLst[idx].item()/16/16/0.45, lossLst[idx].item() + isingExactloss[idx]]

    resultLst = tuple(resultLst)
    print(printString.format(*resultLst))

    heatCapPlot(model, L, batchSize, betaMin=betaMin, betaMax=betaMax, stepNum=num,
                show=False, savePath=loadPath+"pic/", device=device, autodiff=True, loop=loop)
