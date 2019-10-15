import numpy as np
import math

from pybit import bayesian_inference as bi

class SpringModel(bi.Model): 
	""" Class for the spring model """	
	
	def __init__(self, x=[], param=[], name=""): 
	
		# Initialize parent object ModelInference
		bi.Model.__init__(self, name=name)	
	
	def set_param_values(self):

		# Parameters 
		self.C = self.param[0] 
		self.K = self.param[1]
		
		# Variable 
		self.time = self.x 
	
	def fun_x(self):
	
		# Get the parameters
		self.set_param_values() 
		
		return 2 * np.exp(-self.C * self.time / 2) * np.cos(math.sqrt(self.K - self.C**2 / 4) * self.time)