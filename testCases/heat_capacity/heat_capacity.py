from pyBIT import Metropolis_Hastings_Inference as MH

class HeatCapacity(MH.ModelInference): 
	""" 1D Heat conduction model """	
	
	def __init__(self, x=[], param=[]): 
	
		# Initialize parent object ModelInference
		MH.ModelInference.__init__(self)	
		
	def compute_heat_capacity(self):
	
		T = self._x 

		return self._param[0] + self._param[1] * (T / 1000) \
			+ self._param[2] * (T / 1000)**2 + self._param[3] * (T / 1000)**3