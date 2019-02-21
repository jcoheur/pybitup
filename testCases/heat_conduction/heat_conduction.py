import numpy as np
import math

from pyBIT import Metropolis_Hastings_Inference as MH

class HeatConduction(MH.ModelInference): 
	""" 1D Heat conduction model """	
	
	def __init__(self, x=[], param=[]): 
	
		# Initialize parent object ModelInference
		MH.ModelInference.__init__(self)	
		
	def set_param_values(self):
	
		# Parameters
		self.a = self._param[0]
		self.b = self._param[1]
		self.k = self._param[2]
		self.T_amb = self._param[3]
		self.phi = self._param[4]
		self.h = self._param[5]
		
		self.x0 = self._x[0]
		self.L = self._x[-1]
	
	def compute_temperature(self):
	
		# Get the parameters
		self.set_param_values() 
		
		# Compute analytical solution
		gamma = math.sqrt(2 * (self.a + self.b) * self.h / (self.a * self.b * self.k))
		c1 = -self.phi / (self.k * gamma) * (np.exp(gamma * self.L) * (self.h + self.k * gamma) / (np.exp(-gamma * self.L) * (self.h - self.k * gamma) \
			+ np.exp(gamma * self.L) * (self.h + self.k * gamma)))
		c2 = self.phi / (self.k * gamma) + c1
	
		return c1 * np.exp(-gamma * self.x) + c2 * np.exp(gamma * self.x) + self.T_amb
	
	
	
	