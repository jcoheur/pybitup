from scipy import special
import numpy as np
import math	

from pybit import bayesian_inference as bi

class OneReactionPyrolysis(bi.Model): 

    R = 8.314

    def __init__(self, x=[], param=[]): 
            
        # Initialize parent object ModelInference
        bi.Model.__init__(self)
        
        self.xi_init = 0

        self.P = np.array([1.0, 113000, 2.0, 0.04])
                
    def set_param_values(self):

        # Parameters
        self.A = self._param[0] 
        self.E = self._param[1]
        self.n = self._param[2]
        self.m = self._param[3] 
        self.F = self._param[4]
        self.tau = self._param[5]/60 # /!\ tau should be initially in K/m /!\
        
        self.T = self._x 
        self.T_0 = self._x[0]
        
    def solve_system(self): 
        
        # 
        self.set_param_values() 
        
        # Local R 
        R = OneReactionPyrolysis.R 
        
        # Exponential integral function 
        ei = special.expi

        # Analytical solution
        C = (1 - self.xi_init)**(1 - self.n)/(1 - self.n) + (self.A / self.tau) * self.T_0 * np.exp(-self.E / (R * self.T_0)) \
            + ei(-self.E / (R * self.T_0)) * self.E * (self.A / self.tau) / R
        
        self.xi_T = 1 - ((1-self.n) * (-(self.A / self.tau) * self.T * np.exp(-self.E / (R * self.T)) \
            - ei(-self.E / (R * self.T)) * self.E * (self.A / self.tau) / R + C))**(1/(1-self.n))
            
    def compute_gas_prod(self):
        """ Compute gas production""" 
        # Solve the system to get xi_T
        self.solve_system()
        
        # Compute gas production
        gasProd = self.F * (1 - self.xi_T)**self.n * (self.A/self.tau) * self.T**self.m * np.exp(-self.E / (OneReactionPyrolysis.R * self.T))

        return gasProd 

    def fun_x(self): 
        return self.compute_gas_prod()

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

        return X[0]

