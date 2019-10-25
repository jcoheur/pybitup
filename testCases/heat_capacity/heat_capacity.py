from pybitup import bayesian_inference as bi

class HeatCapacity(bi.Model): 
	""" 1D Heat conduction model """	
	
	def __init__(self, x=[], param=[]): 
	
		# Initialize parent object ModelInference
		bi.Model.__init__(self)	
		
	def fun_x(self):
	
		T = self._x 

		return self._param[0] + self._param[1] * (T / 1000) \
			+ self._param[2] * (T / 1000)**2 + self._param[3] * (T / 1000)**3