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
    for i,beta in enumerate(betas):
        for j,cov in enumerate(covs):
            neuronsWithInputs = NeuronsWithInputs(numOfNeurons=2,covariance=cov)
            optimalJSinglePair,MaximalEntropySinglePair =neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
            res[i,j] = optimalJSinglePair
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
    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_different_betas"
    mkdir(f'logs/{dirName}')
    betas = np.arange(0.1,2,0.2)
    # betas = np.arange(0.1,2,1.7)
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    noisyProbs = np.random.rand(cleanProbs.size)
    noisyProbs /= sum(noisyProbs)
    noiseAmounts = np.arange(0,1,0.1)
    # noiseAmounts = np.arange(0,1,1.6)
    res = np.zeros((noiseAmounts.size,betas.size))
    pbar = tqdm.tqdm(total= noiseAmounts.size*betas.size)
    mutinInputsList = np.zeros(len(noiseAmounts))
    for i,noiseAmount in enumerate(noiseAmounts):
        inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbs
        inputProbs /= sum(inputProbs)
        mutinInputs = MutualInformationOfInputs(inputProbs)
        mutinInputsList[i] = mutinInputs
        deltaInMutualInformationNeuronsPerNoise = []
        for j,beta in enumerate(betas):
            neuronsWithInputs = NeuronsWithInputs(numOfNeurons=4,inputProbs=inputProbs)
            optimalJBoth,MaximalEntropyBoth = neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
            inputProbsFirstPair , inputProbsSecondPair = InputSplitter(inputProbs=inputProbs)

            neuronsWithInputsFirst = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsFirstPair)
            neuronsWithInputsSecond = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsSecondPair)
            optimalJFirst,MaximalEntropyFirst = neuronsWithInputsFirst.FindOptimalJPatternSearch(beta=beta)
            optimalJSecond,MaximalEntropySecond = neuronsWithInputsSecond.FindOptimalJPatternSearch(beta=beta)
            deltaInMutualInformationNeuronsPerNoise.append(MaximalEntropyFirst + MaximalEntropySecond - MaximalEntropyBoth)
            res[i,j] = mutinInputs - deltaInMutualInformationNeuronsPerNoise[-1]
            pbar.update(1)
    # plt.plot(betas,MutualInformationOfInputs(inputProbs) -  deltaInMutualInformationNeuronsPerNoise,'o')
    pbar.close()
    fig,ax = plt.subplots()
    ax.imshow(res)
    ax.set_xticks(list(range(len(betas))))
    ax.set_xticklabels(betas)

    # plt.Axes.set_xlim(betas[0],betas[-1])
    ax.set_yticks(list(range(len(mutinInputsList))))
    ax.set_yticklabels(mutinInputsList)
    # plt.Axes.set_yticks(list(range(len(mutinInputs))),mutinInputs)
    # plt.xlabel('Beta')
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
    for i,beta in enumerate(betas):
        effectiveness = np.zeros(noiseAmounts.size)
        print(f"beta = {beta}, ({i},{len(betas)})")
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
    # mutualInformationInputs = np.array(mutualInformationInputs)
    # deltaInMutualInformationNeuronsPerNoise = np.array(deltaInMutualInformationNeuronsPerNoise)
    # mutualInformationInputs = mutualInformationInputs.reshape(deltaInMutualInformationNeuronsPerNoise.shape)
    # pd.DataFrame([{'mutualInformationInputs':mutualInformationInputs,'deltaInMutualInformationNeuronsPerNoise':deltaInMutualInformationNeuronsPerNoise}]).to_csv(f'logs/{dirName}/mutins.csv')

    # plt.plot(mutualInformationInputs - deltaInMutualInformationNeuronsPerNoise,'o')
    # plt.show()
    # plt.title("Difference between mutin of inputs and mutIn of neurons")

def mainSimilarityOfInputs():
    beta = 0.1
    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_beta_{beta}"
    mkdir(f'logs/{dirName}')
    cov1s = np.arange(0,1.1,0.5)
    cov2s = np.arange(0,1.1,0.5)
    deltaInMutualInformation = []
    for covs in tqdm.tqdm(itertools.product(cov1s, cov2s)):
        inputProbs = NoCorrelationInputsBetweenPairs(covs)
        neuronsWithInputs = NeuronsWithInputs(numOfNeurons=4,inputProbs=inputProbs)
        optimalJBoth,MaximalEntropyBoth = neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
        inputProbsFirstPair , inputProbsSecondPair = InputSplitter(inputProbs=inputProbs)

        neuronsWithInputsFirst = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsFirstPair)
        neuronsWithInputsSecond = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsSecondPair)
        optimalJFirst,MaximalEntropyFirst = neuronsWithInputsFirst.FindOptimalJPatternSearch(beta=beta)
        optimalJSecond,MaximalEntropySecond = neuronsWithInputsSecond.FindOptimalJPatternSearch(beta=beta)
        deltaInMutualInformation.append(MaximalEntropyFirst + MaximalEntropySecond - MaximalEntropyBoth)
    deltaInMutualInformation = np.array(deltaInMutualInformation).reshape((cov1s.size,cov2s.size))
    plt.imshow(deltaInMutualInformation)
    plt.show()

def checkingInputCombiner():
    firstPairProbs = Inputs(2,covariance=1).ProbOfAllInputs()
    relationToSecondPair = np.random.rand(firstPairProbs.size,firstPairProbs.size)
    noiseInCorrelation = 1
    plt.plot(InputCombiner(firstPairProbs,relationToSecondPair,noiseInCorrelation))
    plt.show()

def checkingHighBeta():
    betas = np.arange(1,20,1)
    # betas = np.array([9])
    res= np.zeros(betas.size)
    res2= np.zeros(betas.size)
    for i,beta in enumerate(betas):
        covs = [0.1]
        inputProbs = NoCorrelationInputsBetweenPairs(covs)
        neuronsWithInputs = NeuronsWithInputs(numOfNeurons=len(covs)*2,inputProbs=inputProbs)
        optimalJSinglePair,MaximalEntropySinglePair =neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
        res[i] = optimalJSinglePair[0]
        res2[i] = MaximalEntropySinglePair[0]
    plt.plot(res2)
    plt.show()
    #For very high beta, the J barely matters, 
    #as long as it's smaller in size than the input the resulting output will 
    #be the same as the input.   

def checkingInputSplitter():
    # inputProbs = NoCorrelationInputsBetweenPairs([0.5])
    inputProbs = np.array([0,1,0,0])
    print(InputSplitter(inputProbs,[1,1]))

def checkingMutualInformation():
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    noisyProbs = np.random.rand(cleanProbs.size)
    noiseAmounts = np.arange(0,1,0.01)
    mutualInformationInputs = []
    for noiseAmount in tqdm.tqdm(noiseAmounts):
        inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbs
        mutualInformationInputs.append(MutualInformationOfInputs(inputProbs))
    plt.plot(noiseAmounts,mutualInformationInputs)
    plt.show()

def checkingNeuronGroup():
    ngroup = NeuronGroup(2)
    ngroup.H = np.array([1,-1])
    ngroup.J = np.zeros((2,2))
    ngroup.beta = 1
    print(ngroup.ProbOfAllStates())

def checkingJCombiner():
    a = np.zeros((2,2))
    b = np.ones((3,3))
    print(JCombiner(a,b,a))

if __name__ == '__main__':
    # mainDependentInputs()
    # mainSimilarityOfInputs()
    # mainDependentInputsDifferentBetas()
    # mainDifferentInputSameBeta()
    mainDifferentNoise()
    # checkingJCombiner()
    # checkingNeuronGroup()
    # mainIndependentInputs()
    # recreatingResult()
    # mainSimilarityOfInputs()
    # checkingMutualInformation()
    # checkingInputCombiner()
    # checkingInputSplitter()