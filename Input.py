import numpy as np
class Inputs:
    def __init__(self, numNeurons, typeInput = 'binary', covariance = 0,inputProbs=None):
        self.numNeurons = numNeurons
        self.typeInput = typeInput
        self.covariance = covariance
        self.inputProbs = inputProbs
    
    def ProbOfAllInputs(self):
        if self.inputProbs is not None:
            assert sum(self.inputProbs) > 0.95 and sum(self.inputProbs) < 1.05, "sum of given probs doesn't equal 1."
            self.inputProbs /= sum(self.inputProbs)
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
