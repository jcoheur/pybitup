from pybitup import bayesian_inference as bi

class HeatCapacity(bi.Model): 
	""" Model for heat capacity. 
    This test case was implemented for the work of Carlos Garcia Guillamon at VKI. 
    The model for the heat capacity as a function of temperature is the following:
    
    cp(T) = A + B * (T / 1000) + C (T/1000)**2 + D * (T / 1000)**3 + E  / (T / 1000)

    where A, B, C and D and the four unknown model parameters. Nominal parameter values are obtained 
    from optimization : [-333.33, 4095.83, -2553.33, 570.83]. 

    Author: J. Coheur. """	
	
	def __init__(self, x=[], param=[]): 
	
		# Initialize parent object ModelInference
		bi.Model.__init__(self)	
		
	def fun_x(self):
	
		T = self._x 

		return self._param[0] + self._param[1] * (T / 1000) \
			+ self._param[2] * (T / 1000)**2 + self._param[3] * (T / 1000)**3