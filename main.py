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
    betas = np.arange(1,5)
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    noisyProbs = np.random.rand(cleanProbs.size)
    noisyProbs /= sum(noisyProbs)
    noiseAmounts = np.arange(0,1,0.1)
    res = np.zeros((noiseAmounts.size,betas.size))
    for i,noiseAmount in enumerate(noiseAmounts):
        inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbs
        inputProbs /= sum(inputProbs)
        mutinInputs = MutualInformationOfInputs(inputProbs)
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
    # plt.plot(betas,MutualInformationOfInputs(inputProbs) -  deltaInMutualInformationNeuronsPerNoise,'o')
    plt.imshow(res)
    plt.savefig(f'logs/{dirName}/Mutual_information_by_connecting_time_frames_beta_{beta}.png')
    plt.xlabel('Beta')
    plt.ylabel('Mutual Information of pairs of inputs')
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

if __name__ == '__main__':
    # mainDependentInputs()
    # mainSimilarityOfInputs()
    mainDependentInputsDifferentBetas()
    # checkingNeuronGroup()
    # mainIndependentInputs()
    # recreatingResult()
    # mainSimilarityOfInputs()
    # checkingMutualInformation()
    # checkingInputCombiner()
    # checkingInputSplitter()