import numpy as np
from Input import Inputs
from NeuronsWithInputs import NeuronsWithInputs

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
    # assert 0 <= noiseInCorrelation <= 1, f"noiseInCorrelation is supposed to be between 0 and 1, got {noiseInCorrelation}"
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

def MutualInformationOfInputs(inputProbs,sizesOfSplits=[2,2]):
    probsForEachSplit = InputSplitter(inputProbs,sizesOfSplits)
    mutIn = 0
    for input,inputProb in enumerate(inputProbs):
        multOfProbsOfSplits = 1
        for splitInd,splitSize in enumerate(sizesOfSplits):
            multOfProbsOfSplits *= probsForEachSplit[splitInd][(input>>sum(sizesOfSplits[splitInd+1:])) & (2**splitSize - 1)]
        mutIn += inputProb * np.log(inputProb / multOfProbsOfSplits)
    return mutIn

def JCombiner(*args):
    totalLen = 0
    for J in args:
        totalLen += J.shape[0]
    totalJ = np.zeros((totalLen,totalLen))
    curInd = 0
    for J in args:
        nextInd = curInd + J.shape[0]
        totalJ[curInd:nextInd,curInd:nextInd] = J
        curInd = nextInd
    return totalJ
        


def EffectivenessOfConnecting(inputProbs,beta,mutinInputs = None,numOfNeurons=4):
    neuronsWithInputs = NeuronsWithInputs(numOfNeurons=numOfNeurons,inputProbs=inputProbs)
    optimalJBoth,   MaximalEntropyBoth = neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
    inputProbsFirstPair , inputProbsSecondPair = InputSplitter(inputProbs=inputProbs)

    neuronsWithInputsFirst = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsFirstPair)
    neuronsWithInputsSecond = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsSecondPair)
    optimalJFirst,MaximalEntropyFirst = neuronsWithInputsFirst.FindOptimalJPatternSearch(beta=beta)
    optimalJSecond,MaximalEntropySecond = neuronsWithInputsSecond.FindOptimalJPatternSearch(beta=beta)
    
    if mutinInputs is None:
        mutinInputs = MutualInformationOfInputs(inputProbs)
    return mutinInputs + MaximalEntropyBoth - MaximalEntropyFirst - MaximalEntropySecond
    