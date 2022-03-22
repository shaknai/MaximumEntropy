from aifc import Error
from os import stat
from tabnanny import verbose
from tkinter import NO
from urllib.parse import _NetlocResultMixinStr
import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt
from functools import lru_cache
import tqdm
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.factory import Himmelblau
from pymoo.optimize import minimize
from pymoo.core.problem import Problem,ElementwiseProblem

class NeuronGroup:
    def __init__(self,numOfNeurons):
        self.beta = 1
        self.numOfNeurons = numOfNeurons

    def ProbOfState(self,state):
        if isinstance(state,int):
            state = self.NumToBitsArray(state)
        if 0 in state:
            state -= 0.5
            state *= 2
        return np.e ** (self.beta * self.HamiltonianOfState(state))

    def ProbOfAllStates(self):
        res = np.zeros(2**self.numOfNeurons)
        for state in range(2**self.numOfNeurons):
            res[state] = self.ProbOfState(state)
        res /= np.sum(res)
        return res

    def HamiltonianOfState(self, state):
        return (state @ self.H) + (state @ self.J @ state)

    def NumToBitsArray(self,num):
        amountOfBits = self.numOfNeurons
        bits = np.array([])
        while num != 0 and amountOfBits > 0:
            newBits = np.unpackbits(np.array([num & (2 ** min(8,amountOfBits) - 1)],dtype=np.uint8))
            newBits = newBits[-min(8,amountOfBits):]
            bits = np.concatenate([newBits,bits],axis=0)
            num = num >> 8
            amountOfBits -= 8
        if amountOfBits > 0:
            bits = np.concatenate([np.zeros(amountOfBits),bits],axis = 0)
        return bits
   
class Inputs:
    def __init__(self, numNeurons, typeInput = 'binary', covariance = 0,inputProbs=None):
        self.numNeurons = numNeurons
        self.typeInput = typeInput
        self.covariance = covariance
        self.inputProbs = inputProbs
    
    def ProbOfAllInputs(self):
        if self.inputProbs is not None:
            assert sum(self.inputProbs) == 1, "sum of given probs doesn't equal 1."
            return self.inputProbs
        if self.typeInput == 'binary':
            if self.numNeurons == 2:
                probSame = (1+self.covariance)/4
                probDiff = (1-self.covariance)/4
                probs = np.zeros(self.numNeurons**2)
                for i in range(len(probs)):
                    if i in [0,3]:
                        probs[i] = probSame
                    else:
                        probs[i] = probDiff
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return probs

    def InputToH(self,input):
        bits = [x for x in bin(input)[2:]]
        H = np.array([0] * (self.numNeurons - len(bits)) + bits,dtype=np.float32)
        H = (H - 0.5) * 2
        return H


class NeuronsWithInputs:
    def __init__(self,numOfNeurons=2,typeInput='binary',covariance=0,inputProbs=None):
        self.numOfNeurons = numOfNeurons
        self.neuronGroup = NeuronGroup(numOfNeurons)
        self.inputs = Inputs(numOfNeurons,typeInput,covariance,inputProbs)
    
    def oneDJToMat(self,J):
        res = np.zeros((self.numOfNeurons,self.numOfNeurons))
        indInJ=0
        for i in range(self.numOfNeurons):
            for j in range(i+1,self.numOfNeurons):
                res[i,j] = J[indInJ]
                indInJ += 1
        return res

    def ProbOfAllStates(self,J=None,beta=None,covariance = None):
        if covariance:
            self.inputs.covariance = covariance
        
        self.neuronGroup.J = self.oneDJToMat(J)
        
        if beta:
            self.neuronGroup.beta = beta
        probOfAllInputs = self.inputs.ProbOfAllInputs()
        probOfAllStates = np.zeros(2 ** self.neuronGroup.numOfNeurons)
        for input,probOfInput in enumerate(probOfAllInputs):
            self.neuronGroup.H = self.inputs.InputToH(input)
            probOfStatesForInput = self.neuronGroup.ProbOfAllStates()
            probOfAllStates += probOfStatesForInput * probOfInput
        probOfAllStates /= np.sum(probOfAllStates)
        return probOfAllStates
    
    def EntropyOfOutputs(self,J,beta=None,covariance=None):
        if beta:
            self.neuronGroup.beta = beta
        if covariance:
            self.inputs.covariance = covariance
        probOfAllStates = self.ProbOfAllStates(J,beta)
        return entropy(probOfAllStates)

    def NoisyEntropy(self,J,beta=None,covariance=None):
        if covariance:
            self.inputs.covariance = covariance
        if beta:
            self.neuronGroup.beta = beta
        totalEntropy = 0
        probOfAllInputs = self.inputs.ProbOfAllInputs()
        self.neuronGroup.J = self.oneDJToMat(J)
        # self.neuronGroup.J = np.array([[0,J],[0,0]])
        for input,probOfInput in enumerate(probOfAllInputs):
            self.neuronGroup.H = self.inputs.InputToH(input)
            probOfStatesForInput = self.neuronGroup.ProbOfAllStates()
            totalEntropy += probOfInput * entropy(probOfStatesForInput)
        return totalEntropy

    def MutualInformation(self,J,beta=None,covariance=None):
        if beta: 
            self.neuronGroup.beta = beta
        if covariance:
            self.inputs.covariance = covariance
        return self.EntropyOfOutputs(J,beta) - self.NoisyEntropy(J,beta)

    def FindOptimalJBruteForce(self,beta,covariance):
        Js = np.arange(-10,10.05,0.1)
        mutalInformations = np.array([self.MutualInformation(J,beta,covariance) for J in Js])
        bestJ = Js[np.argmax(mutalInformations)]
        # plt.figure()
        # plt.plot(Js,mutalInformations)
        # plt.show(block=False)
        return bestJ
    
    def MinusMutualInformation(self,J):
        return -self.MutualInformation(J)

    def FindOptimalJPatternSearch(self,beta,covariance=0,inputProbs=None):
        self.neuronGroup.beta = beta
        self.inputs.covariance = covariance
        if inputProbs:
            self.inputs.inputProbs = inputProbs    
        problem = ElementWiseMinEntropy(self)
        algorithm = PatternSearch()
        res = minimize(problem,algorithm,verbose=False,seed=1)
        return res.X, -res.F #Returns optimal J and maximal mutual information

    
    def optimalJGradient(self,J):
        delta = 0.1
        return -(self.MutualInformation(J + delta) - self.MutualInformation(J - delta)) / (2*delta)

# class MyProblem(Problem):
#     def __init__(self,neuronsWithInputs):
#         super().__init__(n_var=1, n_obj=1, n_constr=0, xl=-1, xu=1)
#         self.neuronsWithInputs = neuronsWithInputs

#     def _evaluate(self, x, out, *args, **kwargs):
#          out["F"] = np.array([self.neuronsWithInputs.MinusMutualInformation(indX) for indX in x])

    
class ElementWiseMinEntropy(ElementwiseProblem):
    def __init__(self,neuronsWithInputs):
        amountOfEdges = neuronsWithInputs.numOfNeurons * (neuronsWithInputs.numOfNeurons - 1) // 2
        xl = np.zeros(amountOfEdges) - 1
        xu = np.zeros(amountOfEdges) + 1
        super().__init__(n_var = amountOfEdges, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.neuronsWithInputs = neuronsWithInputs      
    def _evaluate(self, x, out, *args, **kwargs):
         out["F"] = np.array(self.neuronsWithInputs.MinusMutualInformation(x))

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



def mainIndependentInputs():
    beta = 0.5
    covs = [0.5]
    inputProbs = NoCorrelationInputsBetweenPairs(covs)
    neuronsWithInputs = NeuronsWithInputs(numOfNeurons=len(covs)*2,inputProbs=inputProbs)
    optimalJSinglePair,MaximalEntropySinglePair =neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
    covs = [0.5,0.5]
    inputProbs = NoCorrelationInputsBetweenPairs(covs)
    neuronsWithInputs = NeuronsWithInputs(numOfNeurons=len(covs)*2,inputProbs=inputProbs)
    optimalJTwoPairs,MaximalEntropyTwoPairs = neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)
    print(f"{MaximalEntropyTwoPairs}, {2*MaximalEntropySinglePair}")

def mainDependentInputs():
    firstPairProbs = NoCorrelationInputsBetweenPairs([0.5])
    relationToSecondPair = np.random.rand(firstPairProbs.size,firstPairProbs.size)
    noiseInCorrelation = 1
    beta = 1
    inputProbs = InputCombiner(firstPairProbs=firstPairProbs,relationToSecondPair=relationToSecondPair,noiseInCorrelation=noiseInCorrelation)
    neuronsWithInputs = NeuronsWithInputs(numOfNeurons=4,inputProbs=inputProbs)
    optimalJBoth,MaximalEntropyBoth = neuronsWithInputs.FindOptimalJPatternSearch(beta=beta)

    print(f"Maximal mutual information of inputs with noise {noiseInCorrelation}: {MaximalEntropyBoth}")
    
def checkingInputCombiner():
    firstPairProbs = Inputs(2,covariance=1).ProbOfAllInputs()
    relationToSecondPair = np.random.rand(firstPairProbs.size,firstPairProbs.size)
    noiseInCorrelation = 1
    plt.plot(InputCombiner(firstPairProbs,relationToSecondPair,noiseInCorrelation))
    plt.show()

def checkingInputSplitter():
    # inputProbs = NoCorrelationInputsBetweenPairs([0.5])
    inputProbs = np.array([0,1,0,0])
    print(InputSplitter(inputProbs,[1,1]))

if __name__ == '__main__':
    # main()
    # mainDependentInputs()
    # checkingInputCombiner()
    checkingInputSplitter()