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
def NoCorrelationInputsBetweenPairs(covs):
    probsOfEachPair = []
    amountOfPairs = len(covs)
    amountOfNeurons = amountOfPairs*2
    for cov in covs:
        probsOfEachPair.append(Inputs(2,covariance=cov).ProbOfAllInputs())
    probOfAllStates = np.ones(2**(amountOfNeurons))
    for state in range(len(probOfAllStates)):
        for pair in range(amountOfPairs):
            probOfAllStates[state] *= probsOfEachPair[pair][(state>>(2*pair)) & 3]
    return probOfAllStates

def InputCombiner(firstPairProbs, relationToSecondPair, noiseInCorrelation = 0):
    assert 0 <= noiseInCorrelation <= 1, f"noiseInCorrelation is supposed to be between 0 and 1, got {noiseInCorrelation}"
    probOfBothInputs = np.zeros(relationToSecondPair.size)
    for input in range(probOfBothInputs.size):
        indexFirstPair = input & 3
        indexSecondPair = input >> 2
        probOfFirstPair = firstPairProbs[indexFirstPair]
        cleanProbBothPairs = probOfFirstPair * relationToSecondPair[indexFirstPair,indexSecondPair]
        noise = np.random.rand() / probOfBothInputs.size
        probOfBothInputs[input] = (1-noiseInCorrelation) * cleanProbBothPairs + noiseInCorrelation*noise
    probOfBothInputs /= np.sum(probOfBothInputs)
    return probOfBothInputs

def InputSplitter(inputProbs,sizesOfSplits=[2,2]):
    probsForEachSplit = [np.zeros(2**size) for size in sizesOfSplits]
    for input,inputProb in enumerate(inputProbs):
        for splitInd,splitSize in enumerate(sizesOfSplits):
            probsForEachSplit[splitInd][(input>>sum(sizesOfSplits[splitInd+1:])) & (2**splitSize - 1)] += inputProb
    return probsForEachSplit

def MutualInfromationOfInputs(inputProbs,sizesOfSplits=[2,2]):
    probsForEachSplit = InputSplitter(inputProbs,sizesOfSplits)
    mutIn = 0
    for input,inputProb in enumerate(inputProbs):
        multOfProbsOfSplits = 1
        for splitInd,splitSize in enumerate(sizesOfSplits):
            multOfProbsOfSplits *= probsForEachSplit[splitInd][(input>>sum(sizesOfSplits[splitInd+1:])) & (2**splitSize - 1)]
        mutIn += inputProb * np.log(inputProb / multOfProbsOfSplits)
    return mutIn

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

def mainDependentInputs():
    # firstPairProbs = NoCorrelationInputsBetweenPairs([0.5])
    # relationToSecondPair = np.random.rand(firstPairProbs.size,firstPairProbs.size)
    # noiseInCorrelation = 1
    beta = 1
    dirName = f"{datetime.now().strftime('%d-%m-%Y_(%H:%M:%S)')}_beta_{beta}"
    mkdir(f'logs/{dirName}')
    cleanProbs = NoCorrelationInputsBetweenPairs([0.5,0.5])
    noisyProbs = np.random.rand(cleanProbs.size)
    noisyProbs /= sum(noisyProbs)
    pd.DataFrame({'cleanProbs':cleanProbs,'noisyProbs':noisyProbs}).to_csv(f'logs/{dirName}/probs.csv')
    noiseAmounts = np.arange(0,1,0.01)
    deltaInMutualInformationNeuronsPerNoise = []
    mutualInformationInputs = []
    for noiseAmount in tqdm.tqdm(noiseAmounts):
        inputProbs = (1-noiseAmount)*cleanProbs + noiseAmount*noisyProbs
        inputProbs /= sum(inputProbs)
        mutualInformationInputs.append(MutualInfromationOfInputs(inputProbs))
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
    plt.savefig(f'logs/{dirName}/Mutual_information_by_connecting_time_frames_beta_{beta}.png')
    plt.show()
    mutualInformationInputs = np.array(mutualInformationInputs)
    deltaInMutualInformationNeuronsPerNoise = np.array(deltaInMutualInformationNeuronsPerNoise)
    mutualInformationInputs = mutualInformationInputs.reshape(deltaInMutualInformationNeuronsPerNoise.shape)
    pd.DataFrame([{'mutualInformationInputs':mutualInformationInputs,'deltaInMutualInformationNeuronsPerNoise':deltaInMutualInformationNeuronsPerNoise}]).to_csv(f'logs/{dirName}/mutins.csv')

    plt.plot(mutualInformationInputs - deltaInMutualInformationNeuronsPerNoise,'o')
    plt.show()
    plt.title("Difference between mutin of inputs and mutIn of neurons")


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
        mutualInformationInputs.append(MutualInfromationOfInputs(inputProbs))
    plt.plot(noiseAmounts,mutualInformationInputs)
    plt.show()

def checkingNeuronGroup():
    ngroup = NeuronGroup(2)
    ngroup.H = np.array([1,-1])
    ngroup.J = np.zeros((2,2))
    ngroup.beta = 1
    print(ngroup.ProbOfAllStates())

if __name__ == '__main__':
    # main()
    # mainDependentInputs()
    checkingHighBeta()
    # checkingNeuronGroup()
    # mainIndependentInputs()
    # recreatingResult()
    # mainSimilarityOfInputs()
    # checkingMutualInformation()
    # checkingInputCombiner()
    # checkingInputSplitter()