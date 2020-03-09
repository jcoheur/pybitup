import numpy as np

from one_reaction_pyrolysis import OneReactionPyrolysis

class OneReactionPyrolysisRescaled(OneReactionPyrolysis):

    def __init__(self): 

        OneReactionPyrolysis.__init__(self)

        self.P = np.array([1.0, 113000, 2.0, 0.04])

    def parametrization_forward(self, X):

        X1 = self.P[1] / (OneReactionPyrolysis.R * 800)

        Y = np.zeros(len(X[:]))

        Y[0] = np.log(X[0]) - X[1] / self.P[1] * X1

        Y[1] =  X[1] / self.P[1]

        Y[2] = X[2] / self.P[2]

        Y[3] = X[3] / self.P[3]

        return Y
        
    def parametrization_backward(self, Y):

        X1 = self.P[1] / (OneReactionPyrolysis.R * 800)
        
        X = np.zeros(len(Y[:]))
        
        X[0] = np.exp(Y[0] + Y[1]*X1)
        
        X[1] = Y[1] * self.P[1]
        
        X[2] = Y[2] * self.P[2]
        
        X[3] = Y[3] * self.P[3]
        
        return X
        
    def parametrization_det_jac(self, X):

        return 1/X[0] 
