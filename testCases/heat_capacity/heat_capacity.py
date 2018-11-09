def model_def(T, param):
	return param[0] + param[1] * (T / 1000) + param[2] * (T / 1000)**2 + param[3] * (T / 1000)**3