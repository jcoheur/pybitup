import matplotlib.pyplot as plt
import pickle
import json 
from jsmin import jsmin
import numpy as np

def post_process_data(input_file_name, caseName):

	# Open and read input file 
	# -------------------------
	# First, remove comments from the file with jsmin because json doesn't allow it
	with open("{}".format(input_file_name)) as js_file:
		minified = jsmin(js_file.read())
	user_inputs = json.loads(minified)
	# Previous solution: if there is no comment in the file
	#with open("heat_capacity.json", 'r') as input_file:
	#	user_inputs = json.load(input_file)
	

	# -------------------------------------------
	# --------- Plot experimental data ----------
	# -------------------------------------------
	
	if user_inputs["PostProcess"]["Data"]["display"] == "yes":
		data_exp = load_exp_data(caseName)
		
		plt.figure(user_inputs["PostProcess"]["Data"]["num_plot"])
		plt.plot(data_exp.x[0,:], data_exp.y[0,:], 'o', mfc='none')
		#, edgecolors='r'
		
		
		
	# -------------------------------------------	
	# --------- Plot initial guess --------------
	# -------------------------------------------
	
	if user_inputs["PostProcess"]["InitialGuess"]["display"] == "yes":
		data_exp = load_exp_data(caseName)
		data_init = np.load("output_{}/fun_eval.{}.npy".format(caseName, 0))	
	
		plt.figure(user_inputs["PostProcess"]["InitialGuess"]["num_plot"])
		plt.plot(data_exp.x[0,:], data_init[0,:])
		
		
	# -------------------------------------------	
	# --------- Plot markov chains --------------
	# -------------------------------------------	
	
	if user_inputs["PostProcess"]["MarkovChain"]["display"] == "yes":
		n_iterations = int(user_inputs['Inference']['algorithm']['n_iterations'])
		n_unpar = len(user_inputs['Inference']['param'])
		param_value = np.zeros((n_iterations+2,n_unpar))
		with open('output_{}/mcmc_chain.dat'.format(caseName), 'r') as file_param:
			i=0
			for line in file_param: 
				c_chain = line.strip()
				param_value[i, :] = np.fromstring(c_chain[1:len(c_chain)-1], sep=' ')
				i+=1
				
		data_exp = load_exp_data(caseName)
		for i in range(n_unpar):
			plt.figure(100+i)
			plt.plot(range(n_iterations+2), param_value[:,i])
			
		
	# -------------------------------------------
	# ------ Posterior predictive check ---------
	# -------------------------------------------
	
	if user_inputs["PostProcess"]["Posterior"]["display"] == "yes":
		plt.figure(user_inputs["PostProcess"]["Posterior"]["num_plot"])	
		
		start_val = int(user_inputs["PostProcess"]["Posterior"]["start_val"])
		delta_it = int(user_inputs["PostProcess"]["Posterior"]["delta_it"])
		end_val = int(user_inputs["PostProcess"]["Posterior"]["end_val"])
		
		data_exp = load_exp_data(caseName)
		data_ij_max=np.zeros((data_exp.n_data_set(), data_exp.size_x(0)))
		data_ij_min=np.zeros((data_exp.n_data_set(), data_exp.size_x(0)))
		for i in range(data_exp.n_data_set()):
	
			# Initialise bounds
			data_i1 = np.load("output_{}/fun_eval.{}.npy".format(caseName, start_val))
			data_ij_max[i,:] = data_i1[0,:]
			data_ij_min[i,:] = data_i1[0,:]
			
			for j in range(start_val+delta_it, end_val+1, delta_it):
				
				# Load current data 
				data_ij = np.load("output_{}/fun_eval.{}.npy".format(caseName, j))
				
				# Update bounds
				for k in range(data_exp.size_x(i)):
					if data_ij_max[i,k] < data_ij[0,k]:
						data_ij_max[i,k] = data_ij[0,k]
					elif data_ij_min[i,k] > data_ij[0,k]:
						data_ij_min[i,k] = data_ij[0,k]
					
				plt.plot(data_exp.x[0,:], data_ij[0,:], alpha=0.0)
			
			plt.fill_between(data_exp.x[0,:], data_ij_min[i,:], data_ij_max[i,:], alpha=0.2)
		
		
	
	# Show plot 
	plt.show()
	
def load_exp_data(caseName): 

	with open('{}_data'.format(caseName), 'rb') as file_data_exp:
		pickler_data_exp = pickle.Unpickler(file_data_exp)
		data_exp = pickler_data_exp.load()
		
	return data_exp