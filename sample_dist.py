import os
import json
from jsmin import jsmin
import pandas as pd
import pickle

import numpy as np

import pybit.distributions 
import pybit.bayesian_inference
import pybit.inference_problem
import pybit.post_process

import matplotlib.pyplot as plt
# Colors
lineColor = [['C0'], ['C1'], ['C2'], [
    'C3'], ['C4'], ['C5'], ['C6'], ['C7']]

class SolveProblem(): 
    

    def __init__(self, input_file_name): 

         # -------------------------
        # Open and read input file 
        # -------------------------

        # First, remove comments from the file with jsmin because json doesn't allow it
        with open("{}".format(input_file_name)) as js_file:
            minified = jsmin(js_file.read())
        self.user_inputs = json.loads(minified)
        # Previous solution: if there is no comment in the file
        #with open("heat_capacity.json", 'r') as input_file:
        #user_inputs = json.load(input_file)

        # Create the output folder
        os.system("mkdir output")


    def sample(self, my_model=[]): 

        if (self.user_inputs.get("Sampling") is not None):
            self.input_sampling = self.user_inputs["Sampling"] 
        else: 
            raise ValueError('Ask for sampling distribution but no Sampling inputs were provided')

        # Set the distribution we want to sample 

        if (self.user_inputs["Sampling"].get("Distribution") is not None):
            # Sample from a known distribution 

            # Get distribution 
            distr_name = []
            distr_param = []
            distr_init_val = []
            distr_name.append(self.user_inputs['Sampling']['Distribution']['name']) 
            distr_param.append(self.user_inputs['Sampling']['Distribution']['hyperparameters'])
            distr_init_val.append(self.user_inputs["Sampling"]["Distribution"]["init_val"])
            sample_dist = pybit.distributions.set_probability_dist(distr_name, distr_param)

            unpar_init_val = np.array(self.user_inputs["Sampling"]["Distribution"]["init_val"])
            
        elif (self.user_inputs["Sampling"].get("BayesianPosterior") is not None):
            # Sample a Bayesian Posterior distribution

            # ----------------------------------------------------------
            # Get from file or generate the whole data set for inference 
            # ----------------------------------------------------------
            
            BP_inputs = self.user_inputs["Sampling"]["BayesianPosterior"]
            n_data_set = len(BP_inputs['Data']) # data that can be reproduced using different models
            for data_set in range(n_data_set):
            
                # Load current data and model properties 
                c_data = BP_inputs['Data'][data_set]
                c_model = BP_inputs['Model'][data_set]
                
                # Model 
                # ------
                if c_model['input_file'] == "None": 
                    #param_names = c_model['param_names']
                    param_nom = np.array(c_model['param_values'])
                    #n_param_model = len(param_nom)

                    my_model.param = param_nom
                else: # Section in construction 
                    # Model is build based on a given input file. 
                    # If an input file is provided, the uncertain parameters are specified within the file. 
                    # The uncertain parameters is identified by the keyword "$", e.g. "$myParam$". 
                    # The input file read by the model is in user_inputs['Model']['input_file']. 
                    a=1 # Nothing happens so far. See in function f_X
                
                # Data 
                # -----
                if c_data['Type'] == "ReadFromFile": 
                    reader = pd.read_csv(c_data['FileName'])
                    x = reader[c_data['xField']].values.T[0,:]		
                    y = reader[c_data['yField']].values.T[0,:]
                    std_y = reader[c_data['sigmaField']].values.T[0,:]
                    dataName = c_data['yField'][0]+"_"+c_data['FileName']
                    
                    my_model.x = x 
                elif c_data['Type'] == "GenerateSynthetic": 	
                    if  c_data['x']['Type'] == "range": 
                        x = np.array(np.arange(c_data['x']['Value'][0], c_data['x']['Value'][1], c_data['x']['Value'][2]))
                    elif  c_data['x']['Type'] == "linspace": 
                        x = np.array(np.linspace(c_data['x']['Value'][0], c_data['x']['Value'][1], c_data['x']['Value'][2]))

                    my_model.x = x 

                    std_y = np.array(c_data['y']['Sigma'])
                    y = np.array(pybit.bayesian_inference.generate_synthetic_data(my_model, c_data['y']['Sigma'], c_data['y']['Error']))
                    dataName = c_data['y']['Name']
                    std_y = np.ones(len(y))*std_y

                else: 
                    raise ValueError("Invalid DataType {}".format(c_data['Type'])) 

                # Initialise data set
                if data_set == 0: 
                    data = pybit.bayesian_inference.Data(dataName, x, y, std_y)
                    
                # When there are more than one data set, add them to previous data
                else: 
                    data.add_data_set(dataName, x, y, std_y)		

            # Write the data in output data file 
            with open('output/data', 'wb') as file_data_exp: 
                pickle.dump(data, file_data_exp)

            # ----------
            # Inference
            # ----------
            
            # Get uncertain parameters 
            # -------------------------
            n_unpar = len(BP_inputs['Prior']['Param'])
            unpar_name = []
            for names in BP_inputs['Prior']['Param'].keys():
                unpar_name.append(names)

            # Get a priori information on the uncertain parameters
            # ----------------------------------------------------
            unpar_init_val = []
            unpar_prior_name = []
            unpar_prior_param = []
            for param_val in BP_inputs['Prior']['Param'].values():
                unpar_init_val.append(param_val['initial_val'])
                unpar_prior_name.append(param_val['prior_name']) 
                unpar_prior_param.append(param_val['prior_param'])
            unpar_init_val = np.array(my_model.parametrization_forward(unpar_init_val))

            # Prior 
            # ------ 
            distr_name = [BP_inputs['Prior']['Distribution']]
            hyperparam = [unpar_prior_name, unpar_prior_param]
            prior_dist = pybit.distributions.set_probability_dist(distr_name, hyperparam)

            # Function evaluation from the model as a function of the uncertain parameters only 
            # ----------------------------------------------------------------------------------			

            def f_X(var_param): 
                model_eval = []
                for data_set in range(n_data_set):
            
                    # Load current data and model properties 
                    #c_data = BP_inputs['Data'][data_set]
                    c_model = BP_inputs['Model'][data_set]
                
                    # Model 
                    # ------
                    if c_model['input_file'] == "None": 
                        param_names = c_model['param_names']
                        param_nom = np.array(c_model['param_values'])
                        n_param_model = len(param_nom)

                        my_model.param = param_nom
                    
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

                            my_model.param = vec_param
                    
                        else: # All parameters are uncertain
                            my_model.param = var_param
            
                        model_eval=np.concatenate((model_eval, my_model.fun_x()))
                        
                    else: # Section in construction 
                        # Model is build based on a given input file. 
                        # If an input file is provided, the uncertain parameters are specified within the file. 
                        # The uncertain parameters is identified by the keyword "$", e.g. "$myParam$". 
                        # The input file read by the model is in user_inputs['Model']['input_file']. 

                        param_nom = np.array(c_model['param_values'])			
                        my_model.param = param_nom  	
                        
                        my_model.x = data.x[data.index_data_set[data_set,0]:data.index_data_set[data_set,1]+1]
                        if data_set == 0:
                            model_eval = my_model.fun_x(c_model['input_file'], unpar_name,var_param)
                        else:
                            model_eval = np.concatenate((model_eval, my_model.fun_x(c_model['input_file'], unpar_name,var_param)))
                        
                return model_eval

            
            # Likelihood 
            # -----------
            likelihood_fun = pybit.bayesian_inference.Likelihood(data, f_X)

            # ----------------------
            # Posterior computation 
            # ----------------------
            sample_dist = pybit.bayesian_inference.BayesianPosterior(prior_dist, likelihood_fun, my_model, unpar_init_val) 


        else:
            raise ValueError('No samling distribution provided or invalid name.')

        # Run sampling of the distribution
        if (self.user_inputs["Sampling"].get('Algorithm') is not None): 
            algo = self.user_inputs["Sampling"]["Algorithm"]
            sampling_dist = pybit.inference_problem.Sampler(sample_dist, algo) 
            sampling_dist.sample(unpar_init_val) 

        # Compute the posterior directly from analytical formula (bayes formula in case of Bayesian inference)
        if (self.user_inputs["Sampling"].get('ComputeAnalyticalDistribution') is not None):
            #if (self.user_inputs["Sampling"]['ComputeAnalyticalDistribution'] == "yes"):
            print("Computing analytical distribution function from formula.")
            if (self.user_inputs["Sampling"]["ComputeAnalyticalDistribution"].get('DistributionSupport') is not None): 
                distr_support = self.user_inputs["Sampling"]["ComputeAnalyticalDistribution"]["DistributionSupport"]
                posterior = sample_dist.compute_density(distr_support)
            else:
                posterior = sample_dist.compute_density()
                            


    def post_process_dist(self):

        if (self.user_inputs.get("PostProcess") is not None):
            self.pp = self.user_inputs["PostProcess"] 
        else: 
            raise ValueError('Ask for post processing data but no inputs were provided')

        pybit.post_process.post_process_data(self.pp)


    def propagate(self, my_model=[]): 

        if (self.user_inputs.get("Propagation") is not None):
            self.input_propagation = self.user_inputs["Propagation"] 
        else: 
            raise ValueError('Ask for uncertainty propagation but no Propagation inputs were provided')


        propagation_inputs = self.user_inputs["Propagation"]
        n_design_exp = len(propagation_inputs["Model"])

        # Get uncertain parameters 
        # -------------------------
        # Uncertain parameters are common to all model provided. 

        unpar_inputs = propagation_inputs["Uncertain_param"]
        n_unpar = len(unpar_inputs)
        unpar = {} # Dictionnary containing uncertain parameters and their value
        for name in unpar_inputs.keys():
            c_unpar_input = unpar_inputs[name]
            if c_unpar_input["filename"] == "None":
                a = 1 # Get from distribution 
            else: # Get the parameter values from the file 
                reader = pd.read_csv(c_unpar_input["filename"]) 
                unpar_value = reader.values[:,c_unpar_input["field"]]

                unpar[name] = unpar_value

        n_sample_param = len(unpar[name])


        # Get the design points
        # -----------------------
        # Get the design points (variable input) for each model provided

        design_points = {} 
        for design_exp in range(n_design_exp):
            c_model = propagation_inputs["Model"][design_exp]
            if (c_model.get("model_id") is not None):
                model_id = c_model["model_id"]
            else: 
                model_id = design_exp
            design_point_input = c_model["design_points"]
            reader = pd.read_csv(design_point_input["filename"])
           
            design_points[model_id] = reader.values[:,design_point_input["field"]]
        model_id_list = design_points.keys()



        # Function evaluation from the model as a function of the uncertain parameters only 
        # ----------------------------------------------------------------------------------			

        def f_X(var_param): 
            model_eval = {}

            # x = reader[c_data['xField']].values.T[0,:]		
            # y = reader[c_data['yField']].values.T[0,:]


            # n_data_set = len(BP_inputs['Data']) # data that can be reproduced using different models
            for design_exp, model_id in enumerate(design_points.keys()):
                my_model.x = design_points[model_id] 
                c_model = propagation_inputs["Model"][design_exp]

                # Model 
                # ------

                if c_model['input_file'] == "None": 
                    param_names = c_model['param_names']
                    param_nom = np.array(c_model['param_values'])
                    n_param_model = len(param_nom)

                    my_model.param = param_nom
                
                    var_param_index = []
                    char_name = " ".join(unpar.keys())
                    for idx, name in enumerate(param_names):
                        is_name = char_name.find(name)
                        if is_name >= 0: 
                            var_param_index.append(idx)

                    if n_unpar < n_param_model: # Only a subset of parameters is uncertain
                        vec_param = param_nom 
                        for n in range(0, n_unpar): 
                            vec_param[var_param_index[n]] = var_param[n]

                        my_model.param = vec_param
                
                    else: # All parameters are uncertain
                        my_model.param = var_param
        
                    #model_eval=np.concatenate((model_eval, my_model.fun_x()))
                    model_eval[model_id] = my_model.fun_x()
                    
                else: # Section in construction 
                    # Model is build based on a given input file. 
                    # If an input file is provided, the uncertain parameters are specified within the file. 
                    # The uncertain parameters is identified by the keyword "$", e.g. "$myParam$". 
                    # The input file read by the model is in user_inputs['Model']['input_file']. 

                    param_nom = np.array(c_model['param_values'])			
                    my_model.param = param_nom  	
                    
                    if design_exp == 0:
                        model_eval = my_model.fun_x(c_model['input_file'], unpar.keys(), var_param)
                    else:
                        model_eval = np.concatenate((model_eval, my_model.fun_x(c_model['input_file'], unpar.keys(),var_param)))
                
            return model_eval


        # Run propagation 
        # ---------------
        # Evaluate the model at the parameter values

        c_param = np.zeros(n_unpar)

        # Initialise bounds
        data_ij_max = {}
        data_ij_min = {}
        data_ij_mean = {}
        data_ij_var = {}
        n_points = {}
        for model_id in model_id_list: 
            n_points[model_id] = len(design_points[model_id])
            data_ij_max[model_id] = -1e5*np.ones(n_points[model_id] )
            data_ij_min[model_id] = 1e5*np.ones(n_points[model_id] )
            data_ij_mean[model_id] = np.zeros(n_points[model_id] )
            data_ij_var[model_id] = np.zeros(n_points[model_id] )

        # Iterate of all the parameter values 
        for i in range(n_sample_param): 

            # Get the parameter values 
            for j, name in enumerate(unpar_inputs.keys()):

                c_param[j] = unpar[name][i]

            # Evaluate the function for the current parameter value
            fun_eval = f_X(c_param)

            for model_id in model_id_list: 
                c_eval = fun_eval[model_id]

                # Update bounds
                for k in range(n_points[model_id]):
                    if data_ij_max[model_id][k] < c_eval[k]:
                        data_ij_max[model_id][k] = c_eval[k]
                    elif data_ij_min[model_id][k] > c_eval[k]:
                        data_ij_min[model_id][k] = c_eval[k]


                # Update mean 
                data_ij_mean[model_id][:] = data_ij_mean[model_id][:] + c_eval[:]

        for i, model_id in enumerate(model_id_list): 

            # Compute mean 
            data_ij_mean[model_id][:] = data_ij_mean[model_id][:]/n_sample_param

            plt.figure(i)
            plt.fill_between(design_points[model_id], data_ij_min[model_id][:],
                            data_ij_max[model_id][:], facecolor=lineColor[i][0], alpha=0.1)
        

        plt.show()

  






        



