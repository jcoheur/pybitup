import numpy as np
import math

def model_def(x, param):
	""" 1D Heat conduction model """	
	
	x0 = x[0,0]
	L = x[0,-1]
	
	a = param[0]
	b = param[1]
	k = param[2]
	T_amb = param[3]
	phi = param[4]
	h = param[5]
	
	gamma = math.sqrt(2 * (a + b) * h / (a * b * k))
	c1 = -phi / (k * gamma) * (np.exp(gamma * L) * (h + k * gamma) / (np.exp(-gamma * L) * (h - k * gamma) + np.exp(gamma * L) * (h + k * gamma)))
	c2 = phi / (k * gamma) + c1
	
	return np.array([c1 * np.exp(-gamma * x[0,:]) + c2 * np.exp(gamma * x[0,:]) + T_amb])