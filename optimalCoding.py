from aifc import Error
from os import stat
import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt
from functools import lru_cache

class NeuronGroup:
    def __init__(self,numOfNeurons):
        self.beta = 1
        self.numOfNeurons = numOfNeurons
        self.J = np.random.randn(numOfNeurons,numOfNeurons)
        # self.J = np.zeros((numOfNeurons,numOfNeurons))
        # self.J[5][3] = 5
        np.fill_diagonal(self.J,0)
        self.H = np.random.randn(numOfNeurons)
        # self.H = np.zeros(numOfNeurons)
        self.Z = self.PartitionFunction()

    def PartitionFunction(self):
        self.Z = 1
        return sum([self.ProbOfState(state) for state in range(2 ** self.numOfNeurons)])
    
    def PartitionFunctionMonteCarlo(self,amountOfRuns):
        self.Z = 1
        return sum([self.ProbOfState(self.NumToBitsArray(state)) for state in np.random.choice(range(2**self.numOfNeurons),amountOfRuns)])

    def ProbOfState(self,state):
        if isinstance(state,int):
            state = self.NumToBitsArray(state)
        return np.e ** (self.beta * self.HamiltonianOfState(state)) / self.Z

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
    
    def MonteCarlo(self,amountOfResults):
        states = range(2 ** self.numOfNeurons)
        probOfStates = [self.ProbOfState(state) for state in states]
        choices = np.random.choice(states,amountOfResults,p=probOfStates)
        return np.array([self.NumToBitsArray(choice) for choice in choices])
    
    def ExpectationOfNeuronsFromDist(self):
        expectation = np.zeros(self.numOfNeurons)
        for state in range(2**self.numOfNeurons):
            bitsArray = self.NumToBitsArray(state)
            expectation += bitsArray * self.ProbOfState(state)
        return expectation

    def ExpectationOfPairsFromDist(self):
        expectation = np.zeros((self.numOfNeurons,self.numOfNeurons))
        for state in range(2**self.numOfNeurons):
            bitsArray = (self.NumToBitsArray(state) - 0.5) * 2 
            expectation += np.einsum('i,j->ij',bitsArray,bitsArray) * self.ProbOfState(state)
        return expectation / 2 + 0.5
    
    def ExpectationsFromMonteCarlo(self,amountOfRuns):
        expectationNeuronsMC = np.zeros(self.numOfNeurons)
        totalProb = 0
        expectationPairsMC = np.zeros((self.numOfNeurons,self.numOfNeurons))
        for bitsArray in np.random.choice(range(2**self.numOfNeurons),amountOfRuns):
            bitsArray = self.NumToBitsArray(bitsArray)
            probOfState = self.ProbOfState(bitsArray)
            expectationNeuronsMC += bitsArray * probOfState
            bitsArray = (bitsArray - 0.5) * 2
            expectationPairsMC += np.einsum('i,j->ij',bitsArray,bitsArray) * probOfState
            totalProb += probOfState
        expectationNeuronsMC /= totalProb
        expectationPairsMC /= totalProb
        expectationPairsMC = expectationPairsMC / 2 + 0.5
        return expectationNeuronsMC , expectationPairsMC

    def UpdateH(self, neuronsExpectations):
        neuronsExpectationsDist = self.ExpectationOfNeuronsFromDist()
        self.H += self.lr * np.log((neuronsExpectations + np.finfo(float).eps) / (neuronsExpectationsDist + np.finfo(float).eps))
        self.Z = self.PartitionFunction()
    
    def UpdateJ(self,pairsExpectations):
        pairsExpectationsDist = self.ExpectationOfPairsFromDist()
        if np.min((pairsExpectations + np.finfo(float).eps) / (pairsExpectationsDist + np.finfo(float).eps)) <= 0 :
            raise Error
        self.J += self.lr * np.log((pairsExpectations + np.finfo(float).eps) / (pairsExpectationsDist + np.finfo(float).eps))
        self.Z = self.PartitionFunction()

    def UpdateHJMonteCarlo(self,neuronsExpectations, pairsExpectations, amountOfRuns):
        expectationNeuronsMC , expectationPairsMC = self.ExpectationsFromMonteCarlo(amountOfRuns)
        self.H += self.lr * np.log((neuronsExpectations + np.finfo(float).eps) / (expectationNeuronsMC + np.finfo(float).eps))
        self.J += self.lr * np.log((pairsExpectations + np.finfo(float).eps) / (expectationPairsMC + np.finfo(float).eps))
        self.Z = self.PartitionFunctionMonteCarlo(amountOfRuns)

    def UpdateParameters(self, neuronsExpectations, pairsExpectations,lr):
        self.lr = lr
        self.UpdateH(neuronsExpectations)
        self.UpdateJ(pairsExpectations)
    
    def UpdateParametersMonteCarlo(self, neuronsExpectations, pairsExpectations, lr, amountOfRuns):
        self.lr = lr
        self.UpdateHJMonteCarlo(neuronsExpectations, pairsExpectations, amountOfRuns)
    
    def GIS(self,neuronsExpectations, pairsExpectations,amountOfRuns = 500,lr = 0.01):
        for _ in range(amountOfRuns):
            self.UpdateParameters(neuronsExpectations,pairsExpectations,lr=lr)
    
    def GradientDescent(self, neuronsExpectations, pairsExpectations,amountOfIters = 500, amountOfRuns = 50,lr = 0.01):
        for _ in range(amountOfIters):
            self.UpdateParametersMonteCarlo(neuronsExpectations, pairsExpectations, lr, amountOfRuns)

class Inputs:
    def __init__(self, numNeurons, typeInput = 'binary', covariance = 0):
        self.numNeurons = numNeurons
        self.typeInput = typeInput
        self.covariance = covariance
    
    def ProbOfAllInputs(self):
        if self.typeInput == 'binary':
            if self.numNeurons == 2:
                probSame = (1+self.covariance)/4
                probDiff = (1-self.covariance)/4
                probs = np.zeros(self.numNeurons**2)
                for i in range(len(probs)):
                    
                    if bin(i)[-1] == bin(i)[-2]:
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
    def __init__(self,numOfNeurons=2,typeInput='binary',covariance=0):
        self.neuronGroup = NeuronGroup(numOfNeurons)
        self.inputs = Inputs(numOfNeurons,typeInput,covariance)
    
    def ProbOfAllStates(self,J,beta):
        probOfAllInputs = self.inputs.ProbOfAllInputs()
        probOfAllStates = np.zeros(2 ** self.neuronGroup.numOfNeurons)
        self.neuronGroup.J = np.array([[1,J],[J,1]])
        self.neuronGroup.beta = beta
        for input,probOfInput in enumerate(probOfAllInputs):
            self.neuronGroup.H = self.inputs.InputToH(input)
            probOfStatesForInput = self.neuronGroup.ProbOfAllStates()
            probOfAllStates += probOfStatesForInput * probOfInput
        return probOfAllStates
    
    def EntropyOfOutputs(self,J):
        probOfAllStates = self.ProbOfAllStates(J)
        return entropy(probOfAllStates)

    def ConditionalEntropy(self,J,beta):
        totalEntropy = 0
        probOfAllInputs = self.inputs.ProbOfAllInputs()
        self.neuronGroup.J = np.array([[1,J],[J,1]])
        self.neuronGroup.beta = beta
        for input,probOfInput in enumerate(probOfAllInputs):
            self.neuronGroup.H = self.inputs.InputToH(input)
            probOfStatesForInput = self.neuronGroup.ProbOfAllStates()
            totalEntropy += probOfInput * entropy(probOfStatesForInput)
        return totalEntropy
    
neuronsWithInputs = NeuronsWithInputs()



