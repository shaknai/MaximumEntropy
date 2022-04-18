import numpy as np
class NeuronGroup:
    def __init__(self,numOfNeurons):
        self.beta = 1
        self.numOfNeurons = numOfNeurons
        self.vHamiltonian = np.vectorize(self.HamiltonianOfState)

    def ProbOfState(self,state):
        return np.e ** (self.beta * self.HamiltonianOfState(state))

    def ProbOfStateFromHamiltonian(self,hamiltonian):
        return np.exp(-self.beta * hamiltonian)

    def ProbOfAllStates(self):
        hamiltonians = self.vHamiltonian(np.arange(2**self.numOfNeurons))
        hamiltonians -= np.max(hamiltonians)
        res = self.ProbOfStateFromHamiltonian(hamiltonians)
        res /= np.sum(res)
        return res

    def HamiltonianOfState(self, state):
        if isinstance(state,int) or isinstance(state,np.int64):
            state = self.NumToBitsArray(state)
        if 0 in state:
            state -= 0.5
            state *= 2
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
