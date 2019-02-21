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
	
    # Load experimental data 
	data_exp = load_exp_data(caseName)
	
	# Colors 
	lineColor = [['C0'], ['C1'], ['C2'], ['C3'], ['C4'], ['C5'], ['C6'], ['C7']]

	# -------------------------------------------
	# --------- Plot experimental data ----------
	# -------------------------------------------
	
	if user_inputs["PostProcess"]["Data"]["display"] == "yes":
		
		for i in range(data_exp.n_data_set):
		
			ind_1 = data_exp.index_data_set[i,0]
			ind_2 = data_exp.index_data_set[i,1]
			
			plt.figure(user_inputs["PostProcess"]["Data"]["num_plot"])
			plt.plot(data_exp.x[ind_1:ind_2+1], data_exp.y[ind_1:ind_2+1], 'o', color=lineColor[i][0], mfc='none')
			#, edgecolors='r'
		
		
		
	# -------------------------------------------	
	# --------- Plot initial guess --------------
	# -------------------------------------------
	
	if user_inputs["PostProcess"]["InitialGuess"]["display"] == "yes":
		data_init = np.load("output_{}/fun_eval.{}.npy".format(caseName, 0))	
	
		for i in range(data_exp.n_data_set):
		
			ind_1 = data_exp.index_data_set[i,0]
			ind_2 = data_exp.index_data_set[i,1]
			
			plt.figure(user_inputs["PostProcess"]["InitialGuess"]["num_plot"])
			plt.plot(data_exp.x[ind_1:ind_2+1], data_init[ind_1:ind_2+1], '--', color=lineColor[i][0])
		
		
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
		
		for i in range(data_exp.n_data_set):
		
			data_ij_max=np.zeros(data_exp.size_x(i))
			data_ij_min=np.zeros(data_exp.size_x(i))
			ind_1 = data_exp.index_data_set[i,0]
			ind_2 = data_exp.index_data_set[i,1]

			# Initialise bounds
			data_i1 = np.load("output_{}/fun_eval.{}.npy".format(caseName, start_val))
			data_ij_max = -1e5*np.ones(data_exp.size_x(i))
			data_ij_min = 1e5*np.ones(data_exp.size_x(i))
			
			for j in range(start_val+delta_it, end_val+1, delta_it):

				# Load current data 
				data_ij = np.load("output_{}/fun_eval.{}.npy".format(caseName, j))
				data_set_n = data_ij[ind_1:ind_2+1]

				# Update bounds
				for k in range(data_exp.size_x(i)):
					if data_ij_max[k] < data_set_n[k]:
						data_ij_max[k] = data_set_n[k]
					elif data_ij_min[k] > data_set_n[k]:
						data_ij_min[k] = data_set_n[k]
				
				plt.plot(data_exp.x[ind_1:ind_2+1], data_set_n[:], alpha=0.0)

			plt.fill_between(data_exp.x[ind_1:ind_2+1], data_ij_min[:], data_ij_max[:], facecolor=lineColor[i][0], alpha=0.1)
			plt.plot(data_exp.x[ind_1:ind_2+1], (data_ij_min+data_ij_max)/2, color=lineColor[i][0], alpha=0.5)
		
			
			del data_ij_max, data_ij_min, data_set_n
		
		
	
	# Show plot 
	plt.show()
	
def load_exp_data(caseName): 

	with open('{}_data'.format(caseName), 'rb') as file_data_exp:
		pickler_data_exp = pickle.Unpickler(file_data_exp)
		data_exp = pickler_data_exp.load()
		
	return data_exp