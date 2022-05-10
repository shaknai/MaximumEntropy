import numpy as np
from Input import Inputs
from NeuronsWithInputs import NeuronsWithInputs
from matplotlib import pyplot as plt
from datetime import datetime
from os import mkdir,system
import pandas as pd
import tqdm
import itertools
from NeuronGroup import NeuronGroup
from utils import *

def recreatingResult():
    betas = np.arange(0.5,2,0.1)
    covs = np.arange(-0.5,0.5,0.1)
    res = np.zeros((betas.size,covs.size))
    pbar = tqdm.tqdm(total=betas.size*covs.size)
    for i,beta in enumerate(betas):
        for j,cov in enumerate(covs):
            neuronsWithInputs = NeuronsWithInputs(numOfNeurons=2,covariance=cov)
            optimalJSinglePair,MaximalEntropySinglePair =neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
            res[i,j] = optimalJSinglePair
            pbar.update(1)
    pbar.close()
    res -= np.min(res) - 1
    plt.imshow(np.log(res))
    plt.show()
   
def mainIndependentInputs():
    beta = 50
    covs = [0.1]
    inputProbs = NoCorrelationInputsBetweenPairs(covs)
    neuronsWithInputs = NeuronsWithInputs(numOfNeurons=len(covs)*2,inputProbs=inputProbs)
    optimalJSinglePair,MaximalEntropySinglePair =neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
    covs = [0.1,0.1]
    inputProbs = NoCorrelationInputsBetweenPairs(covs)
    neuronsWithInputs = NeuronsWithInputs(numOfNeurons=len(covs)*2,inputProbs=inputProbs)
    optimalJTwoPairs,MaximalEntropyTwoPairs = neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
    print(f"{MaximalEntropyTwoPairs}, {2*MaximalEntropySinglePair}")
    print(f"Single pair J: {optimalJSinglePair}")
    print(f"Two pairs J: {optimalJTwoPairs}")

def mainDependentInputsDifferentBetas():
    betas = np.arange(0.1,2,0.2)
    noiseAmounts = np.arange(0,1,0.1)
    numOfNeurons = 4

    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_different_betas"
    mkdir(f'logs/{dirName}')

    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    noisyProbs = np.random.rand(cleanProbs.size)
    noisyProbs /= sum(noisyProbs)
    mutinInputsList = np.zeros(len(noiseAmounts))
    res = np.zeros((noiseAmounts.size,betas.size))

    pbar = tqdm.tqdm(total= noiseAmounts.size*betas.size)

    for i,noiseAmount in enumerate(noiseAmounts):
        inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbs
        inputProbs /= sum(inputProbs)
        mutinInputs = MutualInformationOfInputs(inputProbs)
        mutinInputsList[i] = mutinInputs
        # deltaInMutualInformationNeuronsPerNoise = []
        for j,beta in enumerate(betas):
            #TODO: Use EfficiancyOfInputs here.
            res[i,j] = EffectivenessOfConnecting(inputProbs,beta,numOfNeurons)
            pbar.update(1)
    pbar.close()

    fig,ax = plt.subplots()
    ax.imshow(res)

    ax.set_xticks(list(range(len(betas))))
    ax.set_xticklabels(betas)

    ax.set_yticks(list(range(len(mutinInputsList))))
    ax.set_yticklabels(mutinInputsList)

    ax.set_xlabel('Beta')
    ax.set_ylabel('Mutual Information of pairs of inputs')
    ax.set_title('Effectiveness of Connecting Time Frames')

    plt.savefig(f'logs/{dirName}/Mutual_information_by_connecting_time_frames_beta_{beta}.png')
    plt.show()

def mainDifferentInputSameBeta():
    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_different_inputs"
    mkdir(f'logs/{dirName}')
    fig,ax = plt.subplots()
    betas = np.logspace(-1,1,10)
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    noisyProbs = np.random.rand(cleanProbs.size)
    noisyProbs /= sum(noisyProbs)
    noiseAmounts = np.arange(0,1,0.1)
    for j,beta in enumerate(betas):
        effectiveness = np.zeros(noiseAmounts.size)
        print(f"beta = {beta}, ({j},{len(betas)})")
        pbar = tqdm.tqdm(total= noiseAmounts.size)
        mutinInputsList = np.zeros(len(noiseAmounts))
        for i,noiseAmount in enumerate(noiseAmounts):
            inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbs
            inputProbs /= sum(inputProbs)
            mutinInputs = MutualInformationOfInputs(inputProbs)
            mutinInputsList[i] = mutinInputs
            effectiveness[i] = EffectivenessOfConnecting(inputProbs,beta,mutinInputs=mutinInputs)
            pbar.update(1)
        # plt.plot(betas,MutualInformationOfInputs(inputProbs) -  deltaInMutualInformationNeuronsPerNoise,'o')
        pbar.close()
        ax.plot(mutinInputsList,effectiveness,'o')
    ax.legend(betas)
    ax.set_xlabel('Mutual Information of pairs of inputs')
    ax.set_ylabel('Effectivness of connecting pairs')
    ax.set_title(f'Effectiveness of Connecting Time Frames for different betas')
    plt.savefig(f'logs/{dirName}/Mutual_information_by_connecting_time_frames_beta_{beta}.png')
    plt.show()

def mainDifferentNoise():
    # # evenly sampled time at 200ms intervals
    # t = np.arange(0., 5., 0.2)

    # # red dashes, blue squares and green triangles
    # plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    # plt.show()
    # return
    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_different_noises"
    mkdir(f'logs/{dirName}')
    # fig,ax = plt.subplots()
    beta = 1
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    amountOfRuns = 10
    plt.figure()
    for i in range(amountOfRuns):
        np.random.seed()
        noisyProbs = np.random.rand(cleanProbs.size)
        print(np.sum(noisyProbs))
        noisyProbs /= sum(noisyProbs)
        noiseAmounts = np.arange(0.1,1,0.1) #Changed this to be far from zero mutin.
        effectiveness = np.zeros(noiseAmounts.size)
        print(f"Run {i+1} out of: {amountOfRuns}")
        pbar = tqdm.tqdm(total= noiseAmounts.size)
        mutinInputsList = np.zeros(len(noiseAmounts))
        for i,noiseAmount in enumerate(noiseAmounts):
            inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbs
            inputProbs /= sum(inputProbs)
            mutinInputs = MutualInformationOfInputs(inputProbs)
            mutinInputsList[i] = mutinInputs
            effectiveness[i] = EffectivenessOfConnecting(inputProbs,beta,mutinInputs=mutinInputs)
            pbar.update(1)
        plt.plot(mutinInputsList,effectiveness/mutinInputsList,'o')
        pbar.close()
    # ax.plot(ratios)
    plt.xlabel('Mutual information of inputs')
    plt.ylabel('Effectivness of connecting pairs')
    plt.title(f'Effectiveness of Connecting Time Frames for different noise divided by mutin')
    plt.savefig(f'logs/{dirName}/Mutual_information_by_connecting_time_frames_different_noise.png')
    plt.show()



def mainDependentInputs():
    # firstPairProbs = NoCorrelationInputsBetweenPairs([0.5])
    # relationToSecondPair = np.random.rand(firstPairProbs.size,firstPairProbs.size)
    # noiseInCorrelation = 1
    beta = 10
    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_beta_{beta}"
    # mkdir(f'logs/{dirName}')
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    noisyProbs = np.random.rand(cleanProbs.size)
    noisyProbs /= sum(noisyProbs)
    # pd.DataFrame({'cleanProbs':cleanProbs,'noisyProbs':noisyProbs}).to_csv(f'logs/{dirName}/probs.csv')
    # noiseAmounts = np.arange(0,1,0.1)
    noiseAmounts = np.array([0.8])
    deltaInMutualInformationNeuronsPerNoise = []
    mutualInformationInputs = []
    for noiseAmount in tqdm.tqdm(noiseAmounts):
        inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbs
        inputProbs /= sum(inputProbs)
        mutualInformationInputs.append(MutualInformationOfInputs(inputProbs))
        # inputProbs = InputCombiner(firstPairProbs=firstPairProbs,relationToSecondPair=relationToSecondPair,noiseInCorrelation=noiseInCorrelation)
        neuronsWithInputs = NeuronsWithInputs(numOfNeurons=4,inputProbs=inputProbs)
        optimalJBoth,MaximalEntropyBoth = neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
        inputProbsFirstPair , inputProbsSecondPair = InputSplitter(inputProbs=inputProbs)

        neuronsWithInputsFirst = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsFirstPair)
        neuronsWithInputsSecond = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsSecondPair)
        optimalJFirst,MaximalEntropyFirst = neuronsWithInputsFirst.FindOptimalJPatternSearch(beta=beta)
        optimalJSecond,MaximalEntropySecond = neuronsWithInputsSecond.FindOptimalJPatternSearch(beta=beta)
        deltaInMutualInformationNeuronsPerNoise.append(MaximalEntropyFirst + MaximalEntropySecond - MaximalEntropyBoth)
    plt.plot(mutualInformationInputs,deltaInMutualInformationNeuronsPerNoise,'o')
    plt.title("Mutual infromation gained by connecting two time frames")
    # plt.xscale('log')
    plt.xlabel('Mutal information of inputs between the frames')
    plt.ylabel('Mutual infromation difference gained')
    # plt.savefig(f'logs/{dirName}/Mutual_information_by_connecting_time_frames_beta_{beta}.png')
    plt.show()

def mainSimilarityOfInputs():
    beta = 0.1
    cov1s = np.arange(0,1.1,0.5)
    cov2s = np.arange(0,1.1,0.5)

    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_beta_{beta}"
    mkdir(f'logs/{dirName}')
    effectiveness = []
    for covs in tqdm.tqdm(itertools.product(cov1s, cov2s)):
        inputProbs = NoCorrelationInputsBetweenPairs(covs)
        effectiveness.append(EffectivenessOfConnecting(inputProbs,beta=beta))
    effectiveness = np.array(effectiveness).reshape((cov1s.size,cov2s.size))
    plt.imshow(effectiveness)
    plt.show()

def mainSymmetricAndAntisymmetricNoise():
    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_different_symmetry_of_noise"
    mkdir(f'logs/{dirName}')
    beta = 1
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    amountsOfSymmetry = np.arange(0,1.05,0.1)
    plt.figure()
    noisyProbs = np.random.rand(cleanProbs.size)
    noisyProbs /= sum(noisyProbs)
    for epoch,amountOfSymmetry in enumerate(amountsOfSymmetry):
        noisyProbsAfterSymmetrizing = ContinuousSymmetryOfNoise(noisyProbs,amountOfSymmetry)
        noiseAmounts = np.arange(0.1,1,0.1) #Changed this to be far from zero mutin.
        effectiveness = np.zeros(noiseAmounts.size)
        print(f"Run {epoch+1} out of: {amountsOfSymmetry.size}")
        pbar = tqdm.tqdm(total= noiseAmounts.size)
        mutinInputsList = np.zeros(len(noiseAmounts))
        for i,noiseAmount in enumerate(noiseAmounts):
            inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbsAfterSymmetrizing
            inputProbs /= sum(inputProbs)
            mutinInputs = MutualInformationOfInputs(inputProbs)
            mutinInputsList[i] = mutinInputs
            effectiveness[i] = EffectivenessOfConnecting(inputProbs,beta)
            pbar.update(1)
        plt.plot(mutinInputsList,effectiveness/mutinInputsList,'o')
        pbar.close()
    # ax.plot(ratios)
    plt.legend(amountsOfSymmetry)
    plt.xlabel('Mutual information of inputs')
    plt.ylabel('Effectivness of connecting pairs')
    plt.title(f'Effectiveness of Connecting Time Frames for different-symmetry noise')
    plt.savefig(f'logs/{dirName}/Mutual_information_by_connecting_time_frames_different_symmetry_noise.png')
    plt.show()

def mainSymmetryInPairsNoise():
    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_different_symmetry_of_noise"
    mkdir(f'logs/{dirName}')
    beta = 1
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    amountsOfSymmetry = np.arange(0,1.05,0.1)
    plt.figure()
    noisyProbs = np.random.rand(cleanProbs.size)
    noisyProbs /= sum(noisyProbs)
    for epoch,amountOfSymmetry in enumerate(amountsOfSymmetry):
        noisyProbsAfterSymmetrizing = ContinuousSymmetryInPairs(noisyProbs,amountOfSymmetry)
        noiseAmounts = np.arange(0.1,1,0.1) #Changed this to be far from zero mutin.
        effectiveness = np.zeros(noiseAmounts.size)
        print(f"Run {epoch+1} out of: {amountsOfSymmetry.size}")
        pbar = tqdm.tqdm(total= noiseAmounts.size)
        mutinInputsList = np.zeros(len(noiseAmounts))
        for i,noiseAmount in enumerate(noiseAmounts):
            inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbsAfterSymmetrizing
            inputProbs /= sum(inputProbs)
            mutinInputs = MutualInformationOfInputs(inputProbs)
            mutinInputsList[i] = mutinInputs
            effectiveness[i] = EffectivenessOfConnecting(inputProbs,beta)
            pbar.update(1)
        plt.plot(mutinInputsList,effectiveness/mutinInputsList,'o-')
        pbar.close()
    # ax.plot(ratios)
    plt.legend(amountsOfSymmetry)
    plt.xlabel('Mutual information of inputs')
    plt.ylabel('Effectivness of connecting pairs divided by Mutin of inputs')
    plt.title(f'Effectiveness of Connecting Time Frames for different in-pair symmetry noise')
    plt.savefig(f'logs/{dirName}/Mutual_information_by_connecting_time_frames_different_in_pair_symmetry_noise.png')
    plt.show()

if __name__ == '__main__':
    # mainDependentInputs()
    # mainSimilarityOfInputs()
    # mainDependentInputsDifferentBetas()
    # mainSymmetricAndAntisymmetricNoise()
    mainSymmetryInPairsNoise()
    # mainDifferentInputSameBeta()
    # mainDifferentNoise()
    # checkingJCombiner()
    # checkingNeuronGroup()
    # mainIndependentInputs()
    # recreatingResult()
    # mainSimilarityOfInputs()
    # checkingMutualInformation()
    # checkingInputCombiner()
    # checkingInputSplitter()