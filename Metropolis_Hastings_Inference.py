import os 
import numpy as np
import random
from scipy import linalg, stats
import time 


class ModelInference: 
	"""Class for model inference. 
	All models must have variable parameters called x and model parameters called param """
	
	def __init__(self, x=[], param=[]): 
		
		# Variables 
		self._x = x 
		
		# Parameters
		self._param = param
		
	def _get_param(self):
		"""Method that is called when we want to read the attribute 'param' """
        
		return self._param
		
	def _set_param(self, new_param):
		"""Method that is called when we want to modify the attribute 'param' """

		self._param = new_param

	param = property(_get_param, _set_param)
	
	def _get_x(self):
		"""Method that is called when we want to read the attribute 'x' """
        
		return self._x
		
	def _set_x(self, new_x):
		"""Method that is called when we want to modify the attribute 'x' """

		self._x = new_x

	x = property(_get_x, _set_x)


class Model:
	""" Class defining the model for the inference """
	
	def __init__(self, model, function, function_f = lambda X, P: X, function_b = lambda Y, P: Y, function_det_jac = lambda X: 1, scaling_factors_parametrization = 1, name = ''):
		self.name = name 
		self.model = model
		self.function = function 
		self.parametrization_forward = function_f
		self.parametrization_backward = function_b 
		self.parametrization_det_jac = function_det_jac
		self.P = scaling_factors_parametrization 
		
class DataInference:
	"""Class defining the data in the inference problem"""
	
	def __init__(self, name="", x=np.array([[1]]), y=np.array([[1]]), std_y=np.array([[1]])):
		self.name=list([name])
		self.x=x
		self.y=y
		self.std_y=std_y
		
	def size_x(self, i):
		"""Return the length of the i-th data x"""
		return len(self.x[i,0:])
		
	def n_data_set(self):
		"""Return the number of data sets"""
		return len(self.y[0:,0])
				
	def add_data_set(self, new_name, new_x, new_y, new_std_y):
		"""Add a set of data to the object"""
		self.name.append(new_name) 
		self.x=np.concatenate((self.x, new_x), axis=0)
		self.y=np.concatenate((self.y, new_y), axis=0)
		self.std_y=np.concatenate((self.std_y, new_std_y), axis=0)

def write_tmp_input_file(input_file_name, name_param, value_param): 

	# Check inputs
	n_param = len(name_param)
	if n_param is not len(value_param):
		raise ValueError("Parameter names and values must be of the same length") 

	# Open the input file from which we read the data 
	with open(input_file_name) as json_file:

		# Create the new file where we replace the uncertain variables by their values
		with open("tmp_" + input_file_name, "w") as new_input_file: 
		
			# Read json file line by line
			for num_line, line in enumerate(json_file.readlines()):
			
				ind_1 = is_param = line.find("$")
				new_index = ind_1
				l_line = len(line)
				while is_param >= 0:		
				
					ind_2 = ind_1 + line[ind_1+1:l_line].find("$") + 1
					if ind_2 - ind_1 <= 1: 
						raise ValueError("No parameter name specified in {} line {} \n{}".format(input_file_name, num_line, line))
						
					file_param_name = line[ind_1:ind_2+1]

					# Temporary variables (see later)
					new_name_param = list(name_param)
					new_value_param = list(value_param)	
					
					# Check which uncertain parameters are in the current line 
					for idx, name in enumerate(name_param):
					
						key_name = "$" + name + "$"

						# If the parameter name is in the current line, replace by its value 
						if file_param_name == key_name: 
							
							# Create the new line 
							line = line[0:ind_1-1] + "{}".format(value_param[idx]) + line[ind_2+2:len(line)]

							# Once a parameter name has been found, we don't need to keep tracking it
							# in the remaining lines
							new_name_param.remove(name)
							new_value_param.remove(value_param[idx])
							
							# Update index 
							is_param = line[ind_2+2:l_line].find("$")
							ind_1 = ind_2 + is_param - 3
							
							break
						elif idx < n_param-1: 	
							continue # We go to the next parameter in the list 
						else: # There is the keyword "$" in the line but the parameter is not in the list 
							raise ValueError("There is an extra parameter in the line \n ""{}"" " 
							"but {} is not found in the list.".format(line, line[ind_1+1:ind_2]))
					
					if n_param == 0: 
						# We identified all parameters but we found a "$" in the remaining of the input
						raise ValueError("There is an extra parameter in the line \n ""{}"" " 
						"but {} is not found in the list.".format(line, line[ind_1+1:ind_2]))
					
					# Update parameter lists with only the ones that we still didn't find 
					name_param = new_name_param
					value_param = new_value_param
					n_param = len(name_param)
					 
					
				# Write the new line in the input file 
				new_input_file.write(line)
					
	# Check that all params have been found in the input file 
	if len(name_param) > 0:
		raise ValueError("Parameter(s) {} not found in {}".format(name_param, input_file_name)) 
		
def random_walk_metropolis_hastings(caseName, nIterations, param_init, V, model, prior, data, f_X):
	"""Classical random-walk metropolis hastings algorithm

	Created by: Joffrey Coheur 15-10-18

	algo is a structure that specify the options for the adaptive algorithm.

	parametrization specifies reparametrization functions (foward, backward
	and the determinant of the jacobian). """


	fileID=open("param_output_{}.dat".format(caseName), "w")
    
	n_param = param_init.size
	chain_val = np.zeros((nIterations+2, n_param))
	chain_val[0, 0:] = param_init
	proposal = np.zeros(n_param) 

	# Reparametrization 
	x_parametrization = model.parametrization_forward
	y_parametrization = model.parametrization_backward
	det_jac = model.parametrization_det_jac
	vec_X_parametrized = np.zeros(n_param) 
	Y_parametrized = np.zeros(n_param) 
	
	# Function evaluation
	fun_eval_X = f_X(param_init)
	
	# Save the initial guess '0' as in output file 
	os.system("mkdir output_{}".format(caseName));
	np.save("output_{}/fun_eval.0".format(caseName), fun_eval_X)
	fileID.write("{}\n".format(str(chain_val[0, 0:]))); 
	
	SS_X=sum_of_square(data, fun_eval_X); 
	max_LL=SS_X; # Store the maximum of the log-likelihood function in this variable
	arg_max_LL=param_init;

	# Cholesky decomposition of V. Constant if MCMC is not adaptive
	R = linalg.cholesky(V);
			
	# Monitoring the chain
	n_rejected = 0;

	# Print current time and start clock count
	print("Start time {}" .format(time.asctime(time.localtime())))
	t1 = time.clock()
	
	for i in range(nIterations+1):

		# Guess parameter (candidate or proposal)
		z_k = np.zeros((1,n_param))
		for j in range(n_param):
			z_k[0,j] = random.gauss(0,1)

		proposal[:] = chain_val[i, :] + np.transpose(np.matmul(R, np.transpose(z_k)))
        
		vec_X_parametrized[:] = x_parametrization(chain_val[i, :], model.P)
		Y_parametrized[:] = vec_X_parametrized[:] + np.transpose(np.matmul(R, np.transpose(z_k)))
		proposal[:] = y_parametrization(Y_parametrized[:], model.P)

		
		# Compute ratio of prior distribution 
		pi_0_X = prior.compute_value(chain_val[i, :])
		pi_0_Y = prior.compute_value(proposal[:])
		
		# Test prior values to avoid computation of 0/0 
		if pi_0_Y <= 0: 
			# A new sample out of bounds always rejected
			r = 0
		elif pi_0_X <= 0: 
			# Previous sample out of bounds always make the new one accepted
			# (if it is in the bounds, otherwise it is in the case above)
			r = 1
		else:
			# Acceptance ratio
			fun_eval_Y=f_X(proposal)
			SS_Y=sum_of_square(data, fun_eval_Y)

			# Compare value of SS_Y to get MLE
			if abs(SS_Y) < abs(max_LL):
				max_LL=SS_Y
				arg_max_LL=proposal[:]              			   

			r_det_jac=det_jac(proposal[:])/det_jac(chain_val[i, :]) # Ratio of the determinant of jacobians
			r = np.exp(SS_Y-SS_X)*r_det_jac 
			
			# Multiply by the ratio of prior values
			r_pi_0 = pi_0_Y/pi_0_X # This ratio can be compute safely 
			r *= r_pi_0 
			
		
		alpha = min(1,r);

		# Update 
		u = random.random() # Uniformly distributed number in the interval [0,1)
		if u < alpha: # Accepted
			chain_val[i+1, :] = proposal[:]  
			fun_eval_X=fun_eval_Y
			SS_X = SS_Y
		else: # Rejected
			chain_val[i+1, :] = chain_val[i, :]
			n_rejected+=1

		if i%(nIterations/100) == 0: # We save 100 function evaluation for the post process
			np.save("output_{}/fun_eval.{}".format(caseName, i), fun_eval_X)	   
		
		if i == 100:
			print("Estimated time: {}".format(time.strftime("%H:%M:%S", time.gmtime((time.clock()-t1) / 100.0 * nIterations))))
		
	
		# Write new parameter values
		fileID.write("{}\n".format(str(chain_val[i+1, 0:]))); 

	fileID.close()	
	print("End time {}" .format(time.asctime(time.localtime())))
	print("Elapsed time: {} sec".format(time.strftime("%H:%M:%S", time.gmtime(time.clock()-t1))))
	print("Rejection rate is {} %".format(n_rejected/nIterations*100));
	
class Prior: 
	""" Class Prior contains the a priori information on the parameters 
	prior = Prior(nameList, parameterList)
	nameList contains the names of the marginal prior distribution (there are
	assumed to be independant from each other). 
	parameterList contains the parameter of the corresponding distribution 
	Prior distributions implemented are : Uniform, Gamma, Gaussian

	Uniform:
	--------- 
	Specify lower bounds and upper bounds in param, that must be a
	[2 x dim] vector, e.g. [lowerBounds; upperBounds]

	Gamma: 
	------
	Shape and scale must be specified in a [2 x dim] vector

	Gaussian:
	-------- 
	Mean and standard deviation must be provided in a [2 x dim] vector 
	
	Created by: Joffrey Coheur 17-06-18
	Last modification: Joffrey Coheur 09-11-18"""
		
	def __init__(self, nameList, parameterList):
		self.names = nameList
		self.parameters = parameterList 	

		# Check that the prior function provided in names are implemented 
		implemented_functions = " ".join(["Uniform", "Gamma", "Gaussian"])
		for name in self.names: 
				idx = implemented_functions.find(name)
				if idx < 0: 
					raise ValueError("No {} prior function implemented \n"
					"Implemented prior functions are {}.".format(name,implemented_functions))
		
	def compute_value(self, X): 
		""" compute_prior(X) compute the value of the joint prior at X, where 
		X is a (1 x n_param) numpy array """


		dim_sample = X.size
		Y = 1

		for k in range(dim_sample): 
			priorName = self.names[k]
			param = self.parameters[k]
			
			if priorName == "Uniform":
			
				lb = param[0]
				ub = param[1]

				# Test bounds and compute probability value
				Y = Y * (1/(ub - lb))
				if X[k] < lb or X[k] > ub:
					Y = 0
					# Because there is a zero in the product, we can directly exit the loop
					break 
									
			elif priorName == "Gamma":
			
				theta1 = param[0]
				theta2 = param[1]

				# Compute probability value
				Y = Y * stats.gamma.pdf(X[k], theta1, 0, theta2) 
				
			elif priorName == "Gaussian":
			
				mu = param[0]
				sigma = param[1]

				# Compute probability value
				Y = Y * stats.norm.pdf(X[k], mu, sigma)
				
		return Y 
	
def sum_of_square(data1, data2):

	J = 0
	for i in range(data1.n_data_set()):

		J -=  1/2 * np.sum(((data1.y[i, :]-data2[i, :])/data1.std_y[i, :])**2, axis=0)
		
	return J