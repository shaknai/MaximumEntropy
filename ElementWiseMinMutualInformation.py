import numpy as np
from pymoo.core.problem import Problem,ElementwiseProblem

class ElementWiseMinMutualInformation(ElementwiseProblem):
    def __init__(self,neuronsWithInputs):
        amountOfEdges = neuronsWithInputs.numOfNeurons * (neuronsWithInputs.numOfNeurons - 1) // 2
        xl = np.zeros(amountOfEdges) - 1
        xu = np.zeros(amountOfEdges) + 1
        super().__init__(n_var = amountOfEdges, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.neuronsWithInputs = neuronsWithInputs      
    def _evaluate(self, x, out, *args, **kwargs):
         out["F"] = np.array(self.neuronsWithInputs.MinusMutualInformationNeurons(x))
