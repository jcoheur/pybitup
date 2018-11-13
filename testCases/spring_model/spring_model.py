import numpy as np
import math

from pyBIT import Metropolis_Hastings_Inference as MH

class SpringModel(MH.ModelInference): 
	""" Class for the spring model """	
	
	def __init__(self, x=[], param=[]): 
	
		# Initialize parent object ModelInference
		MH.ModelInference.__init__(self)	
	
	def set_param_values(self):

		# Parameters 
		self.C = self._param[0] 
		self.K = self._param[1]
		
		# Variable 
		self.time = self._x 
	
	def compute_elongation(self):
	
		# Get the parameters
		self.set_param_values() 
		
		return 2 * np.exp(-self.C * self.time / 2) * np.cos(math.sqrt(self.K - self.C**2 / 4) * self.time)