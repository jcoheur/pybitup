import numpy as np
import random
import pickle
import json 
from jsmin import jsmin
import pandas as pd
import os 

import pyBIT.Metropolis_Hastings_Inference as MH

def run_inference(input_file_name, my_model): 

	# Open and read input file 
	# -------------------------
	# First, remove comments from the file with jsmin because json doesn't allow it
	with open("{}".format(input_file_name)) as js_file:
		minified = jsmin(js_file.read())
	user_inputs = json.loads(minified)
	# Previous solution: if there is no comment in the file
	#with open("heat_capacity.json", 'r') as input_file:
	#	user_inputs = json.load(input_file)
	
	# Model 
	# ------
	print("Running \"{}\" model".format(user_inputs['Model']['model_name']))
	if user_inputs['Model']['input_file'] == "None": 
		param_names = user_inputs['Model']['param_names']
		param_nom = np.array(user_inputs['Model']['param_values'])
		n_param_model = len(param_nom)
		
		my_model.model.param = param_nom
	else: # Model is build based on a given input file 
		a = 1
	
	
	# Data 
	# -----
	if user_inputs['Data']['Type'] == "ReadFromFile": 
		reader = pd.read_csv(user_inputs['Data']['FileName'])
		x = reader[user_inputs['Data']['xField']].values.T 
		y = reader[user_inputs['Data']['yField']].values.T
		std_y = reader[user_inputs['Data']['sigmaField']].values.T
		dataName = user_inputs['Data']['yField']
		
		my_model.model.x = x 
	elif user_inputs['Data']['Type'] == "GenerateSynthetic": 	
		if  user_inputs['Data']['x']['Type'] == "range": 
			x = np.array([np.arange(user_inputs['Data']['x']['Value'][0], user_inputs['Data']['x']['Value'][1], user_inputs['Data']['x']['Value'][2])])
		elif  user_inputs['Data']['x']['Type'] == "linspace": 
			x = np.array([np.linspace(user_inputs['Data']['x']['Value'][0], user_inputs['Data']['x']['Value'][1], user_inputs['Data']['x']['Value'][2])])
			
		my_model.model.x = x 
		
		std_y = np.array([[user_inputs['Data']['y']['Sigma']]])
		y = np.array(generate_synthetic_data(my_model, user_inputs['Data']['y']['Sigma'], user_inputs['Data']['y']['Error']))
		dataName = user_inputs['Data']['y']['Name']
		
	else: 
		print("Invalid DataType {}".format(user_inputs['Data']['Type'])) 	
		
	data = MH.DataInference(dataName, x, y, std_y)

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
	
	# Redefine the model as a function of the uncertain parameters only
	# -----------------------------------------------------------------
	var_param_index = []
	char_name = " ".join(unpar_name)
	for idx, name in enumerate(param_names):
		is_name = char_name.find(name)
		if is_name >= 0: 
			var_param_index.append(idx)

	if n_unpar < n_param_model: # Only a subset of parameters is uncertain
		def f_X(var_param):
			vec_param = param_nom 
			for n in range(0, n_unpar): 
				vec_param[var_param_index[n]] = var_param[n]
				
			my_model.model.param = vec_param
			return my_model.function()
	else: # All parameters are uncertain
		def f_X(var_param):
			my_model.model.param = var_param
			return my_model.function()
			
	# Likelihood 
	# -----------
	
	# Algorithm
	# ---------
	algo_name = user_inputs['Inference']['algorithm']['name']
	n_iterations = int(user_inputs['Inference']['algorithm']['n_iterations']) # must be an integer
	
	if algo_name == "RWMH": 
		
		# Proposal
		# ---------
		if user_inputs['Inference']['algorithm']['proposal']['covariance']['type'] == "diag": 
			proposal_cov = np.diag(user_inputs['Inference']['algorithm']['proposal']['covariance']['value'])
		elif user_inputs['Inference']['algorithm']['proposal']['covariance']['type'] == "full":
			proposal_cov = np.array(user_inputs['Inference']['algorithm']['proposal']['covariance']['value']) 
		else: 
			print("Invalid InferenceAlgorithmProposalConvarianceType name {}".format(user_inputs['Inference']['algorithm']['proposal']['covariance']['type']))
			
		# Run 
		MH.random_walk_metropolis_hastings(user_inputs['Model']['model_name'], n_iterations, unpar_init_val, proposal_cov, my_model, prior, data, f_X)

	with open('{}_data'.format(user_inputs['Model']['model_name']), 'wb') as file_data_exp: 
		pickle.dump(data, file_data_exp)


	
def generate_synthetic_data(my_model, std_y, type_pert): 
	"""Generate synthetic data"""

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
		num_data = len(my_model.model.x[0,:])
		rn_data=np.zeros((1, num_data))
		for i in range(0, num_data):
			rn_data[0,i]=random.gauss(0, std_y)

		y = y + rn_data[0,:]

	return y
	
