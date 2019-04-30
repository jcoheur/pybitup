from scipy import special
import numpy as np
import math	

from pyBIT import Metropolis_Hastings_Inference as MH

class OneReactionPyrolysis(MH.ModelInference): 

	R = 8.314
	
	def __init__(self, x=[], param=[]): 
			
		# Initialize parent object ModelInference
		MH.ModelInference.__init__(self)
		
		self.xi_init = 0
				
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
			
		# Solve the system to get xi_T
		self.solve_system()
		
		# Compute gas production
		gasProd = self.F * (1 - self.xi_T)**self.n * (self.A/self.tau) * self.T**self.m * np.exp(-self.E / (OneReactionPyrolysis.R * self.T))
	
		return gasProd 
	
	
	def parametrization_forward(self, X, P):

		X1 = P[1] / (OneReactionPyrolysis.R * 800)

		Y = np.zeros(len(X[:]))

		Y[0] = np.log(X[0]) - X[1] / P[1] * X1

		Y[1] =  X[1] / P[1]

		Y[2] = X[2] / P[2]

		Y[3] = X[3] / P[3]

		return Y
		
	def parametrization_backward(self, Y, P):
			
		X1 = P[1] / (OneReactionPyrolysis.R * 800)
		
		X = np.zeros(len(Y[:]))
		
		X[0] = np.exp(Y[0] + Y[1]*X1)
		
		X[1] = Y[1] * P[1]
		
		X[2] = Y[2] * P[2]
		
		X[3] = Y[3] * P[3]
		
		return X
		
	def det_jac(self, X):
	
		return X[0]
