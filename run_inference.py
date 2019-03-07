import numpy as np
import random
import pickle
import json 
from jsmin import jsmin
import pandas as pd
import os 

import pyBIT.Metropolis_Hastings_Inference as MH

def run_inference(input_file_name, my_model): 

    # -------------------------
    # Open and read input file 
    # -------------------------
	
    # First, remove comments from the file with jsmin because json doesn't allow it
    with open("{}".format(input_file_name)) as js_file:
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

            my_model.model.param = param_nom
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
			
            my_model.model.x = x 
        elif c_data['Type'] == "GenerateSynthetic": 	
            if  c_data['x']['Type'] == "range": 
                x = np.array(np.arange(c_data['x']['Value'][0], c_data['x']['Value'][1], c_data['x']['Value'][2]))
            elif  c_data['x']['Type'] == "linspace": 
                x = np.array(np.linspace(c_data['x']['Value'][0], c_data['x']['Value'][1], c_data['x']['Value'][2]))

            my_model.model.x = x 

            std_y = np.array(c_data['y']['Sigma'])
            y = np.array(generate_synthetic_data(my_model, c_data['y']['Sigma'], c_data['y']['Error']))
            dataName = c_data['y']['Name']

        else: 
            raise ValueError("Invalid DataType {}".format(c_data['Type'])) 

        # Initialise data set
        if data_set == 0: 
            data = MH.DataInference(dataName, x, y, std_y)
			
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
    prior = MH.Prior(unpar_prior_name, unpar_prior_param)

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

                my_model.model.param = param_nom
			
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

                    my_model.model.param = vec_param
			
                else: # All parameters are uncertain
                    my_model.model.param = var_param
	
                model_eval=np.concatenate((model_eval, my_model.function()))
				
            else: # Section in construction 
	            # Model is build based on a given input file. 
	            # If an input file is provided, the uncertain parameters are specified within the file. 
	            # The uncertain parameters is identified by the keyword "$", e.g. "$myParam$". 
	            # The input file read by the model is in user_inputs['Model']['input_file']. 

                param_nom = np.array(c_model['param_values'])			
                my_model.model.param = param_nom  	
                
                my_model.model.x = data.x[data.index_data_set[data_set,0]:data.index_data_set[data_set,1]+1]
                if data_set == 0:
                    model_eval = my_model.function(c_model['input_file'], unpar_name,var_param)
                else:
                    model_eval = np.concatenate((model_eval, my_model.function(c_model['input_file'], unpar_name,var_param)))
				
        return model_eval

	
	# Likelihood 
	# -----------
	
	
	# Algorithm
	# ---------
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
        run_MCMCM = MH.MetropolisHastings(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, my_model, prior, data, f_X)
        run_MCMCM.random_walk_loop()
    elif algo_name == "AMH": 
        print("Using adaptive random-walk Metropolis-Hastings algorithm.")
        starting_it = int(user_inputs['Inference']['algorithm']['AMH']['starting_it'])
        updating_it = int(user_inputs['Inference']['algorithm']['AMH']['updating_it'])
        eps_v = user_inputs['Inference']['algorithm']['AMH']['eps_v']
        run_MCMCM = MH.AdaptiveMetropolisHastings(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, my_model, prior, data, f_X,
                                                  starting_it, updating_it, eps_v)
        run_MCMCM.random_walk_loop()
    elif algo_name == "DR": 
        print("Using delayed-rejection random-walk Metropolis-Hastings algorithm.")
        gamma = user_inputs['Inference']['algorithm']['DR']['gamma']
        run_MCMCM = MH.DelayedRejectionMetropolisHastings(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, my_model, prior, data, f_X,
                                                  gamma)
        run_MCMCM.random_walk_loop()
    elif algo_name == "DRAM": 
        print("Using delayed-rejection adaptive random-walk Metropolis-Hastings algorithm.")
        starting_it = int(user_inputs['Inference']['algorithm']['DRAM']['starting_it'])
        updating_it = int(user_inputs['Inference']['algorithm']['DRAM']['updating_it'])
        eps_v = user_inputs['Inference']['algorithm']['DRAM']['eps_v']
        gamma = user_inputs['Inference']['algorithm']['DRAM']['gamma']
        run_MCMCM = MH.DelayedRejectionAdaptiveMetropolisHastings(user_inputs['Inference']['inferenceProblem'], n_iterations, unpar_init_val, proposal_cov, my_model, prior, data, f_X,
                                                  starting_it, updating_it, eps_v, gamma)
        run_MCMCM.random_walk_loop()
    else:
        raise ValueError('Algorithm "{}" unknown.'.format(algo_name)) 	
    with open('output/data', 'wb') as file_data_exp: 
        pickle.dump(data, file_data_exp)


	
def generate_synthetic_data(my_model, std_y, type_pert): 
    """Generate synthetic data based on the model provided in my_model 
	with standard deviation std_y """

    if type_pert == 'param':
        print("Generate synthetic data based on perturbed parameters")
        num_param = len(my_model.model.param[:])
        rn_param=np.zeros((1, num_param))
        for i in range(0, num_param):
            rn_param[0,i]=random.gauss(0, std_y)

        my_model.model.param = my_model.model.param+rn_param[0,:]
        y = my_model.function()
    else: 
		
        y = my_model.function() 
		
    if type_pert == 'data':
        print("Generate synthetic data based on perturbed nominal solution")
        num_data = len(my_model.model.x[:])
        rn_data=np.zeros((1, num_data))
        for i in range(0, num_data):
            rn_data[0,i]=random.gauss(0, std_y)

        y = y + rn_data[0,:]

    return y
	
