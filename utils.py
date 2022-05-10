import enum
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
    totalAmountOfNeurons = 0
    for J in args:
        amountOfNeurons = int(np.round((1+(1+8*len(J))**0.5)/2))
        totalAmountOfNeurons += amountOfNeurons
    totalJ = np.zeros(totalAmountOfNeurons*(totalAmountOfNeurons - 1)//2)
    amountOfPreviousNeurons = 0
    totalInd = 0
    for J in args:
        currentAmountOfNeurons = int(np.round((1+(1+8*len(J))**0.5)/2))
        firstNeuronInd = 0
        secondNeuronInd = firstNeuronInd + 1
        for connection in J:
            totalJ[totalInd] = connection
            totalInd += 1
            secondNeuronInd += 1
            if secondNeuronInd == currentAmountOfNeurons:
                for _ in range(totalAmountOfNeurons - (amountOfPreviousNeurons + currentAmountOfNeurons)):
                    totalJ[totalInd] = 0
                    totalInd += 1
                firstNeuronInd += 1
                secondNeuronInd = firstNeuronInd + 1
        amountOfPreviousNeurons += currentAmountOfNeurons
        for _ in range(totalAmountOfNeurons - amountOfPreviousNeurons):
            totalJ[totalInd] = 0
            totalInd += 1
    return totalJ
        
def EffectivenessOfConnecting(inputProbs,beta,numOfNeurons=4):
    neuronsWithInputs = NeuronsWithInputs(numOfNeurons=numOfNeurons,inputProbs=inputProbs)
    optimalJBoth,   MaximalEntropyBoth = neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)

    inputProbsFirstPair , inputProbsSecondPair = InputSplitter(inputProbs=inputProbs)
    neuronsWithInputsFirst = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsFirstPair)
    neuronsWithInputsSecond = NeuronsWithInputs(numOfNeurons=2,inputProbs=inputProbsSecondPair)
    optimalJFirst,MaximalEntropyFirst = neuronsWithInputsFirst.FindOptimalJPatternSearch(beta=beta)
    optimalJSecond,MaximalEntropySecond = neuronsWithInputsSecond.FindOptimalJPatternSearch(beta=beta)
    
    optimalJCombined = JCombiner(optimalJFirst,optimalJSecond)
    MutinBoth = neuronsWithInputs.MutualInformationNeurons(optimalJCombined,beta=beta)
    return MaximalEntropyBoth - MutinBoth    

def LittleEndian(num,size):
    return (num >> size) + ((num & (2**size - 1))<<size)

def SymmetrizeNoise(noiseProbs):
    LESize = int(np.log2(noiseProbs.size)/2)
    res = np.zeros(noiseProbs.size)
    for i in range(noiseProbs.size):
        res[i] = (noiseProbs[i] + noiseProbs[LittleEndian(i,LESize)]) / 2
    return res

def ContinuousSymmetryOfNoise(noiseProbs,amountOfSymmetry):
    assert 0 <= amountOfSymmetry <= 1, f"amountOfSymmetry is supposed to be between 0 and 1, got {amountOfSymmetry}."
    return (1 - amountOfSymmetry) * noiseProbs + amountOfSymmetry * SymmetrizeNoise(noiseProbs)

def SymmetrizeNoiseInPairs(noiseProbs):
    LESize = int(np.log2(noiseProbs.size)/2)
    res = np.zeros_like(noiseProbs)
    for i in range(noiseProbs.size):
        twinI = LittleEndian(i & (2**LESize - 1),LESize // 2) + (LittleEndian(i >> LESize, LESize // 2) << LESize)
        res[i] = noiseProbs[i] + noiseProbs[twinI]
    return res

def ContinuousSymmetryInPairs(noiseProbs,amountOfSymmetry):
    assert 0 <= amountOfSymmetry <= 1, f"amountOfSymmetry is supposed to be between 0 and 1, got {amountOfSymmetry}."
    return (1 - amountOfSymmetry) * noiseProbs + amountOfSymmetry * SymmetrizeNoiseInPairs(noiseProbs)
