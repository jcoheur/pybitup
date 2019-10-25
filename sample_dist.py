import os
import json
from jsmin import jsmin
import pandas as pd
import pickle
import time

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

        # Create the output folder
        os.system("mkdir output")

    def get_model_id(self, c_model, num_model):
        """ Get the id of the current model c_model corresponding to num_model.
        c_model is the model specification from the input file."""

        if (c_model.get("model_id") is not None):
            model_id = c_model["model_id"]
        else: 
            model_id = num_model
        return model_id 


    def f_X(self, var_param, model, model_inputs, unpar_name, design_points): 
        """ Define the vector of model evaluation."""
        
        param_names = model_inputs['param_names']
        param_nom = np.array(model_inputs['param_values'])
        model.x = design_points

        if model_inputs['input_file'] == "None": 

            n_param_model = len(param_nom)

            model.param = param_nom
        
            var_param_index = []
            char_name = " ".join(unpar_name)

            n_unpar = len(unpar_name.keys())

            for idx, name in enumerate(param_names):
                is_name = char_name.find(name)
                if is_name >= 0: 
                    var_param_index.append(idx)

            if n_unpar < n_param_model: # Only a subset of parameters is uncertain
                vec_param = param_nom 
                for n in range(0, n_unpar): 
                    vec_param[var_param_index[n]] = var_param[n]

                model.param = vec_param

            else: # All parameters are uncertain
                model.param = var_param

            model_eval=model.fun_x()
            
            
        else: 
            # Model is build based on a given input file. 
            # If an input file is provided, the uncertain parameters are specified within the file. 
            # The uncertain parameters is identified by the keyword "$", e.g. "$myParam$". 
            # The input file read by the model is in user_inputs['Model']['input_file']. 

            model.param = param_nom
        
            #model_eval = np.concatenate((model_eval, my_model.fun_x(c_model['input_file'], unpar_name,var_param)))
            model_eval = model.fun_x(model_inputs['input_file'], unpar_name, var_param)

        return model_eval

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
            # ----------------------------------------------------------
            # Sample a Bayesian Posterior distribution
            # ----------------------------------------------------------
            # We need to define data and models used for sampling the posterior 

            
            # Get the data sets from file or generate it from the model 
            # ----------------------------------------------------------
            BP_inputs = self.user_inputs["Sampling"]["BayesianPosterior"]
            n_data_set = len(BP_inputs['Data']) # data that can be reproduced using different models
            models = {} # Initialise dictionnary of models 
            design_points = {} # Initialise dictionnary of models 
            for data_set in range(n_data_set):
            
                # Load current data and model properties 
                c_data = BP_inputs['Data'][data_set]
                c_model = BP_inputs['Model'][data_set]
            
                model_id = self.get_model_id(c_model, data_set)
                models[model_id] = my_model

                 
                # Model 
                # ------
                #if c_model['input_file'] == "None": 
                #param_names = c_model['param_names']
                param_nom = np.array(c_model['param_values'])
                #n_param_model = len(param_nom)
                my_model.param = param_nom

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

                # When the model comes from external routine, we specify the design points 
                #if c_model['input_file'] != "None": 
                # Design points for the model need to be specified explicitely 
                #models[model_id].x = data.x[data.index_data_set[data_set,0]:data.index_data_set[data_set,1]+1]
                design_points[model_id] = data.x[data.index_data_set[model_id,0]:data.index_data_set[model_id,1]+1]

            # Write the data in output data file 
            with open('output/data', 'wb') as file_data_exp: 
                pickle.dump(data, file_data_exp)

            # ----------
            # Inference
            # ----------

            # Get uncertain parameters 
            # -------------------------
            n_unpar = len(BP_inputs['Prior']['Param'])
            unpar_name = {} #unpar_name = {}
            for names in BP_inputs['Prior']['Param'].keys():
                unpar_name[names] = [] #unpar_name.append(names)

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
            def vec_model_eval(var_param): 
                model_eval = []
                for num_model, model_id in enumerate(models.keys()):
                    model_id_eval = self.f_X(var_param, models[model_id], BP_inputs['Model'][num_model], unpar_name, design_points[model_id])
                    model_eval=np.concatenate((model_eval, model_id_eval))

                return model_eval
            
            # Likelihood 
            # -----------
            likelihood_fun = pybit.bayesian_inference.Likelihood(data, vec_model_eval)

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
        n_points = {}
        models = {}
        for design_exp in range(n_design_exp):
            c_model = propagation_inputs["Model"][design_exp]
            model_id = self.get_model_id(c_model, design_exp)

            design_point_input = c_model["design_points"]
            reader = pd.read_csv(design_point_input["filename"])
           
            design_points[model_id] = reader.values[:,design_point_input["field"]]
            n_points[model_id] = len(design_points[model_id])

            # For now, we only have one "my_model". So all the models[model_id] are linked. 
            # Ideally, should be changed so that we have my_model[model_id] that can thus allow 
            # to have different model 
            models[model_id] = my_model 

        model_id_list = design_points.keys()


        # Function evaluation from the model as a function of the uncertain parameters only 
        # ----------------------------------------------------------------------------------			
        def vec_model_eval(var_param): 
            model_eval = {}
            for num_model, model_id in enumerate(models.keys()):
                model_eval[model_id] = self.f_X(var_param, my_model, propagation_inputs["Model"][num_model], unpar, design_points[model_id])

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
        data_hist = {}
        for model_id in model_id_list: 
            data_ij_max[model_id] = -1e5*np.ones(n_points[model_id] )
            data_ij_min[model_id] = 1e5*np.ones(n_points[model_id] )
            data_ij_mean[model_id] = np.zeros(n_points[model_id] )
            data_ij_var[model_id] = np.zeros(n_points[model_id] )
            data_hist[model_id] = np.zeros([n_sample_param, n_points[model_id]])

        # Initialize output files 
        output_file_model_eval = {}
        for model_id in model_id_list: 
            output_file_model_eval[model_id]=open('output/'+model_id+'_eval.csv','ab')

        # Print current time and start clock count
        print("Start time {}" .format(time.asctime(time.localtime())))
        self.t1 = time.clock()

        # Iterate over all the parameter values 
        for i in range(n_sample_param): 

            # We estimate time after a hundred iterations
            if i == 100:
                print("Estimated time: {}".format(time.strftime("%H:%M:%S",
                                                time.gmtime((time.clock()-self.t1) / 100.0 * n_sample_param))))


            # Get the parameter values 
            for j, name in enumerate(unpar_inputs.keys()):

                c_param[j] = unpar[name][i]

            # Evaluate the function for the current parameter value
            fun_eval = vec_model_eval(c_param)

            for model_id in model_id_list: 
                c_eval = fun_eval[model_id]

                # Store it for later percentile estimatation 
                data_hist[model_id][i, :] = c_eval

                # Write the csv for the function evaluation 
                np.savetxt(output_file_model_eval[model_id], np.array([c_eval]), fmt="%f", delimiter=",")


                # Update bounds
                for k in range(n_points[model_id]):
                    if data_ij_max[model_id][k] < c_eval[k]:
                        data_ij_max[model_id][k] = c_eval[k]
                    elif data_ij_min[model_id][k] > c_eval[k]:
                        data_ij_min[model_id][k] = c_eval[k]


                # Update mean 
                data_ij_mean[model_id][:] = data_ij_mean[model_id][:] + c_eval[:]

        print("End time {}" .format(time.asctime(time.localtime())))
        print("Elapsed time: {} sec".format(time.strftime(
            "%H:%M:%S", time.gmtime(time.clock()-self.t1))))
            
        # Post process and plot 
        for i, model_id in enumerate(model_id_list): 

            # Compute mean 
            data_ij_mean[model_id][:] = data_ij_mean[model_id][:]/n_sample_param

            # Save values in csv format 
            df = pd.DataFrame({"mean" : data_ij_mean[model_id][:], 
                               "lower_bound": data_ij_min[model_id][:], 
                               "upper_bound": data_ij_max[model_id][:]})
            df.to_csv('output/'+model_id+"_interval.csv", index=None)

            df_CI = pd.DataFrame({"CI_lb" : np.percentile(data_hist[model_id], 2.5, axis=0), 
                                  "CI_ub": np.percentile(data_hist[model_id], 97.5, axis=0)})
            df_CI.to_csv('output/'+model_id+"_CI.csv", index=None) 






        



