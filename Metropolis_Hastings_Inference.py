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
	""" Class defining the model function and reparametrization """
	
	def __init__(self, model, function, function_f = lambda X, P: X, function_b = lambda Y, P: Y, function_det_jac = lambda X: 1, scaling_factors_parametrization = 1, name = ''):
		self.name = name 
		self.model = model
		self.function = function 
		self.parametrization_forward = function_f
		self.parametrization_backward = function_b 
		self.parametrization_det_jac = function_det_jac
		self.P = scaling_factors_parametrization 
		
class DataInference:
	"""Class defining the data in the inference problem and contains all the information"""
	
	def __init__(self, name="", x=np.array([1]), y=np.array([1]), std_y=np.array([1])):
		self.name=list([name])
		self.x=x
		self.num_points=np.array([len(self.x)])
		self.y=y
		self.n_data_set=1
		self.std_y=std_y
		self.index_data_set=np.array([[0,len(self.x)-1]])
		
	def size_x(self, i):
		"""Return the length of the i-th data x"""
		return self.num_points[i]	
	
	def add_data_set(self, new_name, new_x, new_y, new_std_y):
		"""Add a set of data to the object"""
		self.name.append(new_name) 
		self.x=np.concatenate((self.x, new_x))
		self.num_points=np.concatenate((self.num_points, np.array([len(new_x)])), axis=0)
		self.y=np.concatenate((self.y, new_y))
		self.std_y=np.concatenate((self.std_y, new_std_y))
		
		# Add new indices array
		last_index = self.index_data_set[self.n_data_set-1,1]+1
		self.index_data_set=np.concatenate((self.index_data_set,np.array([[last_index,last_index+len(new_x)-1]])),axis=0)

		# Increase the number of dataset
		self.n_data_set+=1

class Likelihood: 
    """"Class defining the function and the properties of the likelihood function"""
	
    def __inti__(self, name): 
        self.name=name
    	
    def compute_val(self):
	    """ Compute value of the likelihood at the current point"""
	
    def compute_ratio(self): 
	    """ Compute the ratio of likelihood function"""
		
    def sum_of_square(self):
	    """ Compute the sum of square at the current point"""

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
							
							# Create the new line and unpdate length
							line = line[0:ind_1-1] + "{}".format(value_param[idx]) + line[ind_2+2:len(line)]
							l_line = len(line)
							
							# Once a parameter name has been found, we don't need to keep tracking it
							# in the remaining lines
							new_name_param.remove(name)
							new_value_param.remove(value_param[idx])
							
							# Update index 
							ind_1 = is_param = line.find("$")
							
							
							break
						elif idx < n_param-1: 	
							continue # We go to the next parameter in the list 
						else: # There is the keyword "$" in the line but the parameter is not in the list 
							raise ValueError("There is an extra parameter in the line \n ""{}"" " 
							"but {} is not found in the list.".format(line, line[ind_1+1:ind_2]))
					
					if n_param == 0: 
						# We identified all parameters but we found a "$" in the remaining of the input
						raise ValueError("We identified all parameters but there is an extra parameter in the line \n ""{}"" " 
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

class MetropolisHastings: 

	def __init__(self, caseName, nIterations, param_init, V, model, prior, data, f_X):
		self.caseName = caseName
		self.nIterations = nIterations
		self.param_init = param_init
		self.V = V
		self.model = model
		self.prior = prior 
		self.data = data 
		self.f_X = f_X
	
	    
		# Initialize parameters and functions
		self.n_param = self.param_init.size
		self.current_val = self.param_init
		self.new_val = np.zeros(self.n_param)
		self.current_fun_eval = f_X(self.current_val)
		self.SS_current_fun_eval=sum_of_square(self.data, self.current_fun_eval) 
		self.max_LL=self.SS_current_fun_eval # Store the maximum of the log-likelihood function
		self.arg_max_LL=self.current_val
		
		# Reparametrization 
		self.x_parametrization = self.model.parametrization_forward
		self.y_parametrization = self.model.parametrization_backward
		self.det_jac = self.model.parametrization_det_jac
		self.vec_X_parametrized = np.zeros(self.n_param) 
		self.Y_parametrized = np.zeros(self.n_param) 
				
		# Save the initial guesses in output folder
		os.system("mkdir output_{}".format(self.caseName))
		self.fileID=open("output_{}/mcmc_chain.dat".format(self.caseName), "w")
		np.save("output_{}/fun_eval.0".format(self.caseName), self.current_fun_eval)
		self.write_val(self.current_val)
		
		# Cholesky decomposition of V. Constant if MCMC is not adaptive
		self.R = linalg.cholesky(V)
				
		# Monitoring the chain
		self.n_rejected = 0

		# Print current time and start clock count
		print("Start time {}" .format(time.asctime(time.localtime())))
		self.t1 = time.clock()
		
		
	def random_walk_loop(self): 
	
		for i in range(self.nIterations+1):

			self.compute_new_val()
			self.compute_acceptance_ratio() 
			self.accept_reject()
	
			# We save 100 function evaluation for the post process
			self.write_fun_eval(i, self.nIterations/100, self.current_fun_eval)  

			# We estimate time after a hundred iterations
			if i == 100:
				self.compute_time(self.t1)
		
			# Save the next current value 
			self.write_val(self.current_val)
		
		self.terminate_loop()		
	
	def compute_new_val(self): 
	
		# Guess parameter (candidate or proposal)
		z_k = np.zeros((1, self.n_param))
		for j in range(self.n_param):
			z_k[0,j] = random.gauss(0,1)

		## Without parameterization 
		#proposal[:] = chain_val[i, :] + np.transpose(np.matmul(R, np.transpose(z_k)))
		
		self.vec_X_parametrized[:] = self.x_parametrization(self.current_val, self.model.P)
		self.Y_parametrized[:] = self.vec_X_parametrized[:] + np.transpose(np.matmul(self.R, np.transpose(z_k)))
		self.new_val[:] = self.y_parametrization(self.Y_parametrized[:], self.model.P)
				
	def compute_acceptance_ratio(self): 
		# Compute ratio of prior distribution 
		pi_0_X = self.prior.compute_value(self.current_val[:])
		pi_0_Y = self.prior.compute_value(self.new_val[:])
		
		# Test prior values to avoid computation of 0/0 
		if pi_0_Y <= 0: 
			# A new sample out of bounds always rejected
			self.r = 0
			
		elif pi_0_X <= 0: 
			# Previous sample out of bounds always make the new one accepted
			# (if it is in the bounds, otherwise it is in the case above)
			self.r = 1
			
		else:
			# Acceptance ratio
			self.new_fun_eval = self.f_X(self.new_val)
			self.SS_new_fun_eval = sum_of_square(self.data, self.new_fun_eval)

			# Compare value of SS_Y to get MLE
			if abs(self.SS_new_fun_eval) < abs(self.max_LL):
				self.max_LL = self.SS_new_fun_eval
				self.arg_max_LL = self.new_val[:]            			   

			r_det_jac = self.det_jac(self.new_val[:]) / self.det_jac(self.current_val[:]) # Ratio of the determinant of jacobians
			self.r = np.exp(self.SS_new_fun_eval-self.SS_current_fun_eval) * r_det_jac 

			# Multiply by the ratio of prior values
			r_pi_0 = pi_0_Y / pi_0_X # This ratio can be compute safely 
			self.r *= r_pi_0 

	def accept_reject(self): 
		
		alpha = min(1,self.r);

		# Update 
		u = random.random() # Uniformly distributed number in the interval [0,1)
		if u < alpha: # Accepted
			self.current_val[:] = self.new_val[:]
			self.current_fun_eval = self.new_fun_eval 
			self.SS_current_fun_eval = self.SS_new_fun_eval 
		else: # Rejected, current val remains the same
			self.n_rejected+=1
	
	def terminate_loop(self): 
		self.fileID.close()	
		print("End time {}" .format(time.asctime(time.localtime())))
		print("Elapsed time: {} sec".format(time.strftime("%H:%M:%S", time.gmtime(time.clock()-self.t1))))
		print("Rejection rate is {} %".format(self.n_rejected/self.nIterations*100))
		
	def compute_time(self, t1):
		""" Return the time in H:M:S from time t1 to current clock time """
		
		print("Estimated time: {}".format(time.strftime("%H:%M:%S", time.gmtime((time.clock()-t1) / 100.0 * self.nIterations))))
	
	def write_fun_eval(self, current_it, save_freq, fun_val): 
		if current_it%(save_freq) == 0: 
			np.save("output_{}/fun_eval.{}".format(self.caseName, current_it), fun_val)	
	
	def write_val(self, value):
		# Write the new current val parameter values
		self.fileID.write("{}\n".format(str(value).replace('\n', '')))
		# replace is used to remove the end of lines in the arrays

#class RandomWalk(MetropolisHastings): 

	
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
	
# def sum_of_square(data1, data2):

	# J = 0
	# for i in range(data1.n_data_set()):

		# J -=  1/2 * np.sum(((data1.y[i, :]-data2[i, :])/data1.std_y[i, :])**2, axis=0)
		
	# return J
	
def sum_of_square(data1, data2):

    J =  - 1/2 * np.sum(((data1.y-data2)/data1.std_y)**2, axis=0)

    return J