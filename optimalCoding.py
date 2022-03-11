from aifc import Error
from os import stat
import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt
from functools import lru_cache
import tqdm

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
        if covariance:
            self.inputs.covariance = covariance
        if beta:
            self.neuronGroup.beta = beta
        probOfAllInputs = self.inputs.ProbOfAllInputs()
        self.neuronGroup.J = np.array([[0,J],[0,0]])
        probOfStates = np.zeros(2**self.neuronGroup.numOfNeurons)
        for input,probOfInput in enumerate(probOfAllInputs):
            self.neuronGroup.H = self.inputs.InputToH(input)
            probOfStatesForInput = self.neuronGroup.ProbOfAllStates()
            probOfStates += probOfInput * probOfStatesForInput
        return entropy(probOfStates)
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

    def MutualInformation(self,J,beta,covariance=None):
        if covariance:
            self.inputs.covariance = covariance
        return self.EntropyOfOutputs(J,beta) - self.NoisyEntropy(J,beta)

    def FindOptimalJ(self,beta,covariance):
        self.inputs.covariance = covariance
        Js = np.arange(-100,100,1)
        noisyEntropys = [self.NoisyEntropy(J,beta) for J in Js]
        plt.plot(Js,noisyEntropys)
        plt.show()
        while True:
            noisyEntropy = self.NoisyEntropy(J,beta)
            noisyEntropyAtDelta = self.NoisyEntropy(J + delta,beta)
            J += lr * (noisyEntropyAtDelta - noisyEntropy)
            loops += 1
            if np.abs((noisyEntropyAtDelta - noisyEntropy) / delta) < 0.001:
                break

        return J

neuronsWithInput = NeuronsWithInputs(covariance=0.5)
alpha = -0.9
beta = 2.5
betas = np.arange(0.5,2.01,0.1)
alphas = np.arange(-0.3,0.31,0.1)
Js = np.arange(-10,10.05,0.1)
optimalJs = np.zeros((betas.size,alphas.size))
for i,beta in tqdm.tqdm(enumerate(betas)):
    for j,alpha in enumerate(alphas):
        mutalInformations = np.array([neuronsWithInput.MutualInformation(J,beta,alpha) for J in Js])
        bestJ = Js[np.argmax(mutalInformations)]
        bestInf = np.max(mutalInformations)
        optimalJs[i,j] = bestJ
im = plt.imshow(optimalJs,extent=[min(alphas),max(alphas),max(betas),min(betas)],cmap=plt.get_cmap('seismic'))
plt.xlabel('Covariance')
plt.ylabel('beta')
plt.title('Optimal J')
plt.colorbar(im)
plt.show()
# plt.plot(Js,mutalInformations)
# plt.plot(bestJ,bestInf,'o')
# plt.show()


# neurons = NeuronGroup(2)
# neurons.H = np.array([1,-1])
# neurons.J = np.zeros((2,2))
# neurons.beta = 1
# print(neurons.ProbOfAllStates())
# neurons.beta = 100
# print(neurons.ProbOfAllStates())
# neurons.J = np.array([[0,1],[0,0]])
# print(neurons.ProbOfAllStates())