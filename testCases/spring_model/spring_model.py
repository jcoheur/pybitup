import numpy as np
import math

def model_def(time, param):
	"""
	C = param[0] 
	K = param[1]
	"""
	C = param[0] 
	K = param[1]

	return 2 * np.exp(-C * time / 2) * np.cos(math.sqrt(K - C**2 / 4) * time)