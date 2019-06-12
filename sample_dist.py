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

        # Save the initial guesses in output folder
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
            n_data_set = len(BP_inputs['Data'])
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
                    a=1 # Nothing happens so far
                
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






        



