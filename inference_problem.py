import numpy as np
from scipy import stats
import random
import pickle
import json
from jsmin import jsmin
import pandas as pd
import os

import pybit.metropolis_hastings_algorithms as mha


class Data:
    """Class defining the data in the inference problem and contains all the information"""

    """     def __init__(self, name="", x=np.array([]), y=np.array([]), n_data_set=0, std_y=np.array([]), index_data_set=np.array([[]])):   
        self.name = list([name])
        self.x = x
        self.num_points = np.array([len(self.x)])
        self.y = y
        self.n_data_set = n_data_set
        self.std_y = std_y
        self.index_data_set = index_data_set """


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
        self.x = np.concatenate((self.x, new_x))
        self.num_points = np.concatenate(
            (self.num_points, np.array([len(new_x)])), axis=0)
        self.y = np.concatenate((self.y, new_y))
        self.std_y = np.concatenate((self.std_y, new_std_y))

        # Add new indices array
        last_index = self.index_data_set[self.n_data_set-1, 1]+1
        self.index_data_set = np.concatenate((self.index_data_set, np.array(
            [[last_index, last_index+len(new_x)-1]])), axis=0)

        # Increase the number of dataset
        self.n_data_set += 1


class Model:
    """ Class defining the model function and its reparametrization if specified."""

    def __init__(self, x=[], param=[], scaling_factors_parametrization=1, name=""):
        self.name = name
        self.P = scaling_factors_parametrization
        
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


    def fun_x(self, val_x, val_param): 

        return 1

    def parametrization_forward(self, X=1, P=1):
        
        Y = X

        return Y  

    def parametrization_backward(self, Y=1, P=1):
        
        X = Y

        return X

    def parametrization_det_jac(self, X=1):

        det_jac = 1 

        return det_jac

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

class Likelihood: 
    """"Class defining the function and the properties of the likelihood function. Not yet defined"""
	
    def __inti__(self, name): 
        self.name=name
    	
    def compute_val(self):
	    """ Compute value of the likelihood at the current point"""
	
    def compute_ratio(self): 
	    """ Compute the ratio of likelihood function"""
		
    def sum_of_square(self):
	    """ Compute the sum of square at the current point"""



class Posterior:
    """ Class defining the posterior distribution function 
    The posterior distribution can be computed within this class using: 
    - Bayes formula 
    - Iterative algorithms.
    """
    def __init__(self, input_file_name, my_model):
        self.input_file_name = input_file_name
        self.my_model = my_model

    def run_inference(self): 

        # -------------------------
        # Open and read input file 
        # -------------------------
        
        # First, remove comments from the file with jsmin because json doesn't allow it
        with open("{}".format(self.input_file_name)) as js_file:
            minified = jsmin(js_file.read())
        user_inputs = json.loads(minified)
        # Previous solution: if there is no comment in the file
        #with open("heat_capacity.json", 'r') as input_file:
        #user_inputs = json.load(input_file)

        # ----------------------------------------------------------
        # Get from file or generate the whole data set for inference 
        # ----------------------------------------------------------
        
        n_data_set = len(user_inputs['Data'])
        for data_set in range(n_data_set):
        
            # Load current data and model properties 
            c_data = user_inputs['Data'][data_set]
            c_model = user_inputs['Model'][data_set]
            
            # Model 
            # ------
            if c_model['input_file'] == "None": 
                param_names = c_model['param_names']
                param_nom = np.array(c_model['param_values'])
                n_param_model = len(param_nom)

                self.my_model.param = param_nom
            else: # Section in construction 
                # Model is build based on a given input file. 
                # If an input file is provided, the uncertain parameters are specified within the file. 
                # The uncertain parameters is identified by the keyword "$", e.g. "$myParam$". 
                # The input file read by the model is in user_inputs['Model']['input_file']. 
                a=1 # Nothing happens so far
            
            # Data 
            # -----
            if c_data['Type'] == "ReadFromFile": 
                reader = pd.read_csv(c_data['FileName'])
                x = reader[c_data['xField']].values.T[0,:]		
                y = reader[c_data['yField']].values.T[0,:]
                std_y = reader[c_data['sigmaField']].values.T[0,:]
                dataName = c_data['yField'][0]+"_"+c_data['FileName']
                
                self.my_model.x = x 
            elif c_data['Type'] == "GenerateSynthetic": 	
                if  c_data['x']['Type'] == "range": 
                    x = np.array(np.arange(c_data['x']['Value'][0], c_data['x']['Value'][1], c_data['x']['Value'][2]))
                elif  c_data['x']['Type'] == "linspace": 
                    x = np.array(np.linspace(c_data['x']['Value'][0], c_data['x']['Value'][1], c_data['x']['Value'][2]))

                self.my_model.x = x 

                std_y = np.array(c_data['y']['Sigma'])
                y = np.array(generate_synthetic_data(self.my_model, c_data['y']['Sigma'], c_data['y']['Error']))
                dataName = c_data['y']['Name']

            else: 
                raise ValueError("Invalid DataType {}".format(c_data['Type'])) 

            # Initialise data set
            if data_set == 0: 
                data = Data(dataName, x, y, std_y)
                
            # When there are more than one data set, add them to previous data
            else: 
                data.add_data_set(dataName, x, y, std_y)		

        # ----------
        # Inference
        # ----------
        
        # Get uncertain parameters 
        # -------------------------
        n_unpar = len(user_inputs['Inference']['param'])
        unpar_name = []
        for names in user_inputs['Inference']['param'].keys():
            unpar_name.append(names)

        # Get a priori information on the uncertain parameters
        # ----------------------------------------------------
        unpar_init_val = []
        unpar_prior_name = []
        unpar_prior_param = []
        for param_val in user_inputs['Inference']['param'].values():
            unpar_init_val.append(param_val['initial_val'])
            unpar_prior_name.append(param_val['prior_name']) 
            unpar_prior_param.append(param_val['prior_param'])
        unpar_init_val = np.array(unpar_init_val)
        
        # Prior 
        # ------ 
        prior = Prior(unpar_prior_name, unpar_prior_param)

        # Function evaluation from the model as a function of the uncertain parameters only 
        # ----------------------------------------------------------------------------------			

        def f_X(var_param): 
            model_eval = []
            for data_set in range(n_data_set):
        
                # Load current data and model properties 
                c_data = user_inputs['Data'][data_set]
                c_model = user_inputs['Model'][data_set]
            
                # Model 
                # ------
                if c_model['input_file'] == "None": 
                    param_names = c_model['param_names']
                    param_nom = np.array(c_model['param_values'])
                    n_param_model = len(param_nom)

                    self.my_model.param = param_nom
                
                    var_param_index = []
                    char_name = " ".join(unpar_name)
                    for idx, name in enumerate(param_names):
                        is_name = char_name.find(name)
                        if is_name >= 0: 
                            var_param_index.append(idx)

                    if n_unpar < n_param_model: # Only a subset of parameters is uncertain
                        vec_param = param_nom 
                        for n in range(0, n_unpar): 
                            vec_param[var_param_index[n]] = var_param[n]

                        self.my_model.param = vec_param
                
                    else: # All parameters are uncertain
                        self.my_model.param = var_param
        
                    model_eval=np.concatenate((model_eval, self.my_model.fun_x()))
                    
                else: # Section in construction 
                    # Model is build based on a given input file. 
                    # If an input file is provided, the uncertain parameters are specified within the file. 
                    # The uncertain parameters is identified by the keyword "$", e.g. "$myParam$". 
                    # The input file read by the model is in user_inputs['Model']['input_file']. 

                    param_nom = np.array(c_model['param_values'])			
                    self.my_model.param = param_nom  	
                    
                    self.my_model.x = data.x[data.index_data_set[data_set,0]:data.index_data_set[data_set,1]+1]
                    if data_set == 0:
                        model_eval = self.my_model.fun_x(c_model['input_file'], unpar_name,var_param)
                    else:
                        model_eval = np.concatenate((model_eval, self.my_model.fun_x(c_model['input_file'], unpar_name,var_param)))
                    
            return model_eval

        
        # Likelihood 
        # -----------
        
        
        # Algorithm
        # ---------
        if user_inputs['Inference']['algorithm'] == "None": 
            #Compute the posterior directly using Bayes formula 
            print("Computing posterior distribution function from Bayes formula.")
            
            posterior = mha.bayes_formula
        else: 
            algo_name = user_inputs['Inference']['algorithm']['name']
            n_iterations = int(user_inputs['Inference']['algorithm']['n_iterations']) # must be an integer

            # Proposal
            # ---------
            if user_inputs['Inference']['algorithm']['proposal']['covariance']['type'] == "diag": 
                proposal_cov = np.diag(user_inputs['Inference']['algorithm']['proposal']['covariance']['value'])
            elif user_inputs['Inference']['algorithm']['proposal']['covariance']['type'] == "full":
                proposal_cov = np.array(user_inputs['Inference']['algorithm']['proposal']['covariance']['value']) 
            else: 
                print("Invalid InferenceAlgorithmProposalConvarianceType name {}".format(user_inputs['Inference']['algorithm']['proposal']['covariance']['type']))

            if algo_name == "RWMH": 
                print("Using random-walk Metropolis-Hastings algorithm.")
                run_MCMCM = mha.MetropolisHastings(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, self.my_model, prior, data, f_X) 
                run_MCMCM.random_walk_loop() 
            elif algo_name == "AMH": 
                print("Using adaptive random-walk Metropolis-Hastings algorithm.")
                starting_it = int(user_inputs['Inference']['algorithm']['AMH']['starting_it'])
                updating_it = int(user_inputs['Inference']['algorithm']['AMH']['updating_it'])
                eps_v = user_inputs['Inference']['algorithm']['AMH']['eps_v']
                run_MCMCM = mha.AdaptiveMetropolisHastings(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, self.my_model, prior, data, f_X,
                                                        starting_it, updating_it, eps_v)
                run_MCMCM.random_walk_loop()
            elif algo_name == "DR": 
                print("Using delayed-rejection random-walk Metropolis-Hastings algorithm.")
                gamma = user_inputs['Inference']['algorithm']['DR']['gamma']
                run_MCMCM = mha.DelayedRejectionMetropolisHastings(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, self.my_model, prior, data, f_X,
                                                        gamma)
                run_MCMCM.random_walk_loop()
            elif algo_name == "DRAM": 
                print("Using delayed-rejection adaptive random-walk Metropolis-Hastings algorithm.")
                starting_it = int(user_inputs['Inference']['algorithm']['DRAM']['starting_it'])
                updating_it = int(user_inputs['Inference']['algorithm']['DRAM']['updating_it'])
                eps_v = user_inputs['Inference']['algorithm']['DRAM']['eps_v']
                gamma = user_inputs['Inference']['algorithm']['DRAM']['gamma']
                run_MCMCM = mha.DelayedRejectionAdaptiveMetropolisHastings(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, self.my_model, prior, data, f_X,
                                                        starting_it, updating_it, eps_v, gamma)
                run_MCMCM.random_walk_loop()
            elif algo_name == "Ito-SDE": 
                print("Running Ito-SDE algorithm.")
                run_MCMCM = mha.ito_SDE(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, self.my_model, prior, data, f_X)
                run_MCMCM.random_walk_loop()
            else:
                raise ValueError('Algorithm "{}" unknown.'.format(algo_name)) 

        with open('output/data', 'wb') as file_data_exp: 
            pickle.dump(data, file_data_exp)


def generate_synthetic_data(my_model, std_y, type_pert):
    """ Generate synthetic data based on the model provided in my_model
    with a given standard deviation std_y """

    if type_pert == 'param':
        print("Generate synthetic data based on perturbed parameters")
        num_param = len(my_model.param[:])
        rn_param = np.zeros((1, num_param))
        for i in range(0, num_param):
            rn_param[0, i] = random.gauss(0, std_y)

        my_model.param = my_model.param+rn_param[0, :]
        y = my_model.fun_x()
    else:
        y = my_model.fun_x()

    if type_pert == 'data':
        print("Generate synthetic data based on perturbed nominal solution")
        num_data = len(my_model.x[:])
        rn_data = np.zeros(num_data)
    for i in range(0, num_data):
        rn_data[i] = random.gauss(0, std_y)

    y = y + rn_data[:]

    return y

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
