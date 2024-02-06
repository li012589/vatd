import torch
import numpy as np
import matplotlib.pyplot as plt
from flow import TpwLinearSpline
import source, flow, utils
import h5py, os
import utils


def heatCapPlot(model, L, batchSize, target, smallBatchNum=1, betaMin=0.5, betaMax=1.5, stepNum=100, savePath=None, show=False, autodiff=True, repAcptRej=False, autodiffLevel=0, device=torch.device('cpu')):
    if savePath is None:
        savePath="./"
    if autodiffLevel == 0:
        savePath += "_level0"
    elif autodiffLevel == 1:
        savePath += "_level1"
    else:
        savePath += "_stat"
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    smallBatch = batchSize // smallBatchNum

    lnZlst = []
    lnZerrlst = []
    heatClst = []
    meanElst = []

    if autodiff or repAcptRej:
        dlogPlst = []

    if autodiff:
        deltaMeanlst = []

    betaLst = torch.linspace(betaMin, betaMax, stepNum).to(device)
    xy = target

    for beta in betaLst:
        print(beta.item())
        for _b in range(smallBatchNum):
            if repAcptRej:
                _beta = beta.detach().repeat(smallBatch, 1).requires_grad_(True).to(device)
                T = 1/_beta

                z = model.prior.sample(smallBatch, T=T)
                if autodiffLevel == 1:
                    z = z.detach()
                zlogProb = model.prior.logProbability(z, T=T)
                samples, logDet = model.forward(z, T=T)
                logProb = zlogProb - logDet
                energyWithT = xy.energyWithT(samples, T=T)
                logZ = -(logProb + energyWithT)

                dlogP = torch.autograd.grad(logProb, beta, grad_outputs=torch.ones(smallBatch, 1).to(device), create_graph=True, retain_graph=True)[0]
                dEdT = torch.autograd.grad(energyWithT, beta, grad_outputs=torch.ones(smallBatch, 1).to(device), create_graph=True, retain_graph=True)[0]

                dlogZ = -dlogP - dEdT

                if autodiffLevel == 0:
                    logQdivR = model.prior.sourceList[1].logAceptRej()

                    dlogQdivR = torch.autograd.grad(logQdivR, beta, grad_outputs=torch.ones(smallBatch, 1).to(device), create_graph=True)[0]

                    with torch.no_grad():
                        B2 = (logZ * (dlogQdivR**2)).mean() / (dlogQdivR**2).mean()
                        B3 = (logProb * (dlogQdivR**2)).mean() / (dlogQdivR**2).mean()

                        partFn = logZ.mean()
                        partFnErr = logZ.std()
                        meanE = -(dlogZ + (logZ - B2) * dlogQdivR).mean()
                        dlogPest = (dlogP + (logProb - B3) * dlogQdivR).mean()

                if autodiffLevel == 1:
                    dzlogProb = torch.autograd.grad(zlogProb, beta, grad_outputs=torch.ones(smallBatch, 1).to(device), create_graph=True, retain_graph=True)[0]

                    with torch.no_grad():
                        B2 = (logZ * (dzlogProb**2)).mean() / (dzlogProb**2).mean()
                        B3 = (logProb * (dzlogProb**2)).mean() / (dzlogProb**2).mean()

                        partFn = logZ.mean()
                        partFnErr = logZ.std()
                        meanE = -(dlogZ + (logZ - B2) * dzlogProb).mean()
                        dlogPest = (dlogP + (logProb - B3) * dzlogProb).mean()

                lnZlst.append(partFn.item()/L**2)
                lnZerrlst.append(partFnErr.item()/L**2)
                meanElst.append(meanE.item()/L**2)

                dlogPlst.append(dlogP.mean().item()/L**2)
                heatClst.append((beta[0] * dlogPest).item()/L**2)

            elif autodiff:
                beta = beta.detach().requires_grad_(True).to(device)

                z = model.prior.sample(smallBatch, T=1/beta)
                z = z.detach()
                zlogProb = model.prior.logProbability(z, T=1/beta)
                samples, logDet = model.forward(z, T=1/beta)
                logProb = zlogProb - logDet
                energyWithT = xy.energyWithT(samples, T=1/beta)
                energy = xy.energyWithT(samples)

                dlogP = torch.autograd.grad(logProb.mean(), beta, create_graph=True, retain_graph=True)[0]

                dEdT = torch.autograd.grad(energyWithT.mean(), beta, create_graph=True, retain_graph=True)[0]

                loss = logProb + energyWithT
                logZ = -(loss).mean()
                dlogZ = -dlogP - dEdT

                deltaMean = -dlogZ - energy.mean()

                with torch.no_grad():
                    partFn = logZ
                    partFnErr = loss.std()
                    meanE = -dlogZ

                lnZlst.append(partFn.item()/L**2)
                lnZerrlst.append(partFnErr.item()/L**2)
                meanElst.append(meanE.item()/L**2)
                deltaMeanlst.append(deltaMean.item()/L**2)

                dlogPlst.append(dlogP.item()/L**2)
                heatClst.append((beta * dlogP.mean()).item()/L**2)

            else:
                # direct sample and statistical
                beta = beta.detach()
                T = 1/beta

                with torch.no_grad():
                    z = model.prior.sample(smallBatch, T=T)
                    zlogProb = model.prior.logProbability(z, T=T)
                    samples, logDet = model.forward(z, T=T)
                    logProb = zlogProb - logDet
                    energy = xy.energy(samples)
                    logZ = -(logProb + energy * beta)
                    partFnErr = logZ.std()
                    logZ = logZ.mean()
                meanE = energy.mean() / L**2
                Cv = beta**2 * ((energy**2).mean() - (energy.mean())**2) / L**2

                lnZlst.append(logZ.item()/L**2)
                lnZerrlst.append(partFnErr.item()/L**2)
                meanElst.append(meanE.item())
                heatClst.append(Cv.item())

    lnZlst = np.array(lnZlst).reshape(stepNum, smallBatchNum).mean(-1)
    meanElst = np.array(meanElst).reshape(stepNum, smallBatchNum).mean(-1)
    heatClst = np.array(heatClst).reshape(stepNum, smallBatchNum).mean(-1)

    if autodiff or repAcptRej:
        dlogPlst = np.array(dlogPlst).reshape(stepNum, smallBatchNum).mean(-1)

    lnZerrlst = np.array(lnZerrlst).reshape(stepNum, smallBatchNum).mean(-1)
    if autodiff:
        deltaMeanlst = np.array(deltaMeanlst).reshape(stepNum, smallBatchNum).mean(-1)

    betaLst = betaLst.cpu().numpy()

    with open(os.path.join(savePath, "xyplotdatalite.npy"), "wb") as f:
        np.save(f, betaLst)
        np.save(f, lnZlst)
        np.save(f, meanElst)
        np.save(f, heatClst)
        np.save(f, dlogPlst)

    plt.figure(figsize=(12, 9))
    plt.title('lnZ')
    plt.xlabel('beta')
    plt.ylabel('lnZ')
    plt.errorbar(betaLst, lnZlst, yerr=lnZerrlst, marker='o', label="est.")
    plt.legend(loc='best')
    plt.savefig(os.path.join(savePath, 'xylnZ.pdf'))

    plt.figure(figsize=(12, 9))
    plt.title('Mean Energy')
    plt.xlabel('beta')
    plt.ylabel('Mean Energy')
    plt.scatter(betaLst, meanElst, marker='o', label="est.")
    plt.legend(loc='best')
    plt.savefig(os.path.join(savePath, 'xymeanE.pdf'))

    if autodiff or repAcptRej:
        plt.figure(figsize=(12, 9))
        plt.title('dlogP')
        plt.xlabel('beta')
        plt.ylabel('dlogP')
        plt.scatter(betaLst, dlogPlst, marker='o', label="est.")
        plt.legend(loc='best')
        plt.savefig(os.path.join(savePath, 'dlogP.pdf'))

    plt.figure(figsize=(12, 9))
    plt.title('Heat Capacity')
    plt.xlabel('beta')
    plt.ylabel('Heat Capacity')
    plt.scatter(betaLst, heatClst, marker='o', label="est.")
    plt.legend(loc='best')
    plt.savefig(os.path.join(savePath, 'xyheatC.pdf'))

    if autodiff:
        plt.figure(figsize=(12, 9))
        plt.title('deltaMean')
        plt.xlabel('beta')
        plt.ylabel('dlogZ')
        plt.scatter(betaLst, deltaMeanlst, marker='o', label="est.")
        plt.legend(loc='best')
        plt.savefig(os.path.join(savePath, 'deltaMean.pdf'))

    if show:
        plt.show()


if __name__ == "__main__":
    #torch.manual_seed(42)
    # parameters for dists
    L = 16  # lattice length
    num = 300 # plot points number
    batchSize = 1000 # the batch size to estimate free energy
    smallBatchNum = 10 # num of runs to get the full batch

    device = torch.device("cpu")
    #device = torch.device("cuda:1")

    loadPath = "./opt/uniformSoftplusCNN_testcubic_vonMises_Multi_8_XY_J1.0_T1.0_L8_t_0.5_0.75_0.85_0.9_1.0_1.1_1.25_2.4/"

    model = torch.load(os.path.join(loadPath, "best_TrainLoss_joint.saving"), map_location=device)

    T0 = 1. # base t for XY dist
    e = -1
    trainTime = -1
    factorLst = [0.5, 0.75, 0.85, 0.9, 1., 1.1, 1.25, 2.4] # multi factors for T0 to form temperatures points
    Tlist = [term * T0 for term in factorLst]
    target = source.XYbreakSym(L=L, d=2, T=T0, J=1.0, flg2d=True, fixFirst=0).to(device)
    lossLst = []
    for t in Tlist:
        with torch.no_grad():
            T = torch.tensor(t).to(device)
            samples, logProb = model.sample(batchSize, T=T)
            loss = (logProb + target.energyWithT(samples, T=T)).mean(dim=0, keepdim=True)
        lossLst.append(loss.detach().item())

    lossLst = np.array(lossLst)
    lossSum = lossLst.sum()
    printString = "epoch: %d, L: %.5f, "
    resultLst = [e, lossSum]
    for idx, factor in enumerate(factorLst):
        printString += "XY@" + str(factor) + ": %.2f, "
        resultLst += [lossLst[idx].item()]
    printString += "time: %.2f"
    resultLst += [trainTime]
    resultLst = tuple(resultLst)
    print(printString % resultLst)

    heatCapPlot(model, L, batchSize, target, stepNum=num, show=True, savePath=os.path.join(loadPath, "pic"), autodiff=False, repAcptRej=True, autodiffLevel=1, device=device)
