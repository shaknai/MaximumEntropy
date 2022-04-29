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
    a = [1,2,3,4,5,6]
    b = [7]
    c = [8]
    print(JCombiner(a,b,c))

if __name__ == '__main__':
    checkingJCombiner()
    # checkingNeuronGroup()
    # checkingMutualInformation()
    # checkingInputCombiner()
    # checkingInputSplitter()