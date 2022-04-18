from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.optimize import minimize
from NeuronGroup import NeuronGroup
from Input import Inputs
import numpy as np
from scipy.stats import entropy
from ElementWiseMinMutualInformation import ElementWiseMinMutualInformation

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

    def MutualInformationNeurons(self,J,beta=None,covariance=None):
        if beta: 
            self.neuronGroup.beta = beta
        if covariance:
            self.inputs.covariance = covariance
        return self.EntropyOfOutputs(J,beta) - self.NoisyEntropy(J,beta)

    def FindOptimalJBruteForce(self,beta,covariance):
        Js = np.arange(-10,10.05,0.1)
        mutalInformations = np.array([self.MutualInformationNeurons(J,beta,covariance) for J in Js])
        bestJ = Js[np.argmax(mutalInformations)]
        # plt.figure()
        # plt.plot(Js,mutalInformations)
        # plt.show(block=False)
        return bestJ
    
    def MinusMutualInformationNeurons(self,J):
        return -self.MutualInformationNeurons(J)

    def FindOptimalJPatternSearch(self,beta,covariance=None,inputProbs=None):
        if beta:
            self.neuronGroup.beta = beta
        if inputProbs:
            self.inputs.inputProbs = inputProbs 
        if covariance:
            self.inputs.covariance = covariance
        problem = ElementWiseMinMutualInformation(self)
        algorithm = PatternSearch()
        res = minimize(problem,algorithm,verbose=False,seed=1)
        return res.X, -res.F #Returns optimal J and maximal mutual information

    
    def optimalJGradient(self,J):
        delta = 0.1
        return -(self.MutualInformationNeurons(J + delta) - self.MutualInformationNeurons(J - delta)) / (2*delta)

