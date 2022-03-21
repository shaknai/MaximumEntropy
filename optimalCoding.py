from aifc import Error
from os import stat
from tabnanny import verbose
import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt
from functools import lru_cache
import tqdm
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.factory import Himmelblau
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

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
                    if i in [0,3]:
                        probs[i] = probSame
                    else:
                        probs[i] = probDiff
            else:
                raise NotImplementedError
        else:
            if self.typeInput == 'random':
                probs = np.random.rand(2**self.numNeurons)
                probs /= sum(probs)
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
    
    def ProbOfAllStates(self,J=None,beta=None,covariance = None):
        if covariance:
            self.inputs.covariance = covariance
        if J:
            self.neuronGroup.J = np.array([[0,J],[0,0]])
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
        self.neuronGroup.J = np.array([[0,J],[0,0]])
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

    def FindOptimalJ(self,beta,covariance):
        Js = np.arange(-10,10.05,0.1)
        mutalInformations = np.array([self.MutualInformation(J,beta,covariance) for J in Js])
        bestJ = Js[np.argmax(mutalInformations)]
        # plt.figure()
        # plt.plot(Js,mutalInformations)
        # plt.show(block=False)
        return bestJ
    
    def MinusMutualInformation(self,J):
        return -self.MutualInformation(J)
    def FindOptimalJGradientDescent(self,beta,covariance,lr,last=None,ratiolrIfClose = 1):
        self.inputs.covariance = covariance
        self.neuronGroup.beta = beta
        if last is None:
            last = 0
        # else:
        #     lr *= ratiolrIfClose
        problem = MyProblem(self)
        algorithm = PatternSearch()
        return minimize(problem,algorithm,verbose=False,seed=1).X
        # return gradient_descent(self.optimalJGradient, last, lr)

    
    def optimalJGradient(self,J):
        delta = 0.1
        return -(self.MutualInformation(J + delta) - self.MutualInformation(J - delta)) / (2*delta)

class MyProblem(Problem):
    def __init__(self,neuronsWithInputs):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=-1, xu=1)
        self.neuronsWithInputs = neuronsWithInputs

    def _evaluate(self, x, out, *args, **kwargs):
         out["F"] = np.array([self.neuronsWithInputs.MinusMutualInformation(indX) for indX in x])

        
# def gradient_descent(gradient, start, learn_rate, n_iter=100, tolerance=1e-06):
#     vector = start
#     for i_iter in range(n_iter):
#         # plt.axvline(vector)
#         diff = -learn_rate * gradient(vector)
#         if np.all(np.abs(diff) <= tolerance):
#             # print(f'Stopped at iter {i_iter}')
#             break
#         vector += diff
        
#     return vector

neuronsWithInput = NeuronsWithInputs(covariance=0.5)
alpha = -0.9
beta = 2.5
betas = np.arange(0.5,2.01,0.01)
alphas = np.arange(-0.3,0.31,0.01)
lr = 1
# beta = -2
# alpha = -0.263
# print(neuronsWithInput.FindOptimalJ(beta,alpha))
# print(neuronsWithInput.FindOptimalJGradientDescent(beta,alpha,lr))
# bla
print("Working")
optimalJs = np.zeros((betas.size,alphas.size))
for i,beta in tqdm.tqdm(enumerate(betas)):
    last = None
    for j,alpha in enumerate(alphas):
        # print(neuronsWithInput.FindOptimalJ(beta,alpha))
        last = neuronsWithInput.FindOptimalJGradientDescent(beta,alpha,lr,last,50)
        optimalJs[i,j] = last
        # print(neuronsWithInput.FindOptimalJGradientDescent(beta,alpha,10))
im = plt.imshow(optimalJs,extent=[min(alphas),max(alphas),max(betas),min(betas)],cmap=plt.get_cmap('seismic'))
plt.xlabel('Covariance')
plt.ylabel('beta')
plt.title('Optimal J')
plt.colorbar(im)
# plt.savefig('Binary_Input_Optimal_J.png')
plt.show()