import os
import json
from jsmin import jsmin
import pandas as pd
import pickle
import time
import sys
import shutil 

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np

import pybitup.distributions 
import pybitup.bayesian_inference
import pybitup.inference_problem
import pybitup.post_process

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

        # Define the Input-Output utilities 
        #----------------------------------

        # Initialise
        self.IO_path = {}
        self.IO_fileID = {}

        # Define the file names 
        path = os.getcwd()
        print("The current working directory is "+path)
        self.IO_path['out_folder'] = path+"/output"
        self.IO_path['out_data'] = self.IO_path['out_folder']+"/output.dat"
        
        # Create the output folder
        try:
            os.mkdir(self.IO_path['out_folder'])
        except OSError:
            print(self.IO_path['out_folder']+" already exists.")
        else:
            print("Creating output directory "+self.IO_path['out_folder'])

        # Output file 
        # 'a+': append mode with creation if does not exists  
        self.IO_fileID['out_data'] = open(self.IO_path['out_data'], 'a+')

         # Copy input file in the output folder to keep track of it 
        print("Copying input file in output folder ... ")
        shutil.copy(input_file_name, self.IO_path['out_folder'])

    def create_output_file(self, IO_path_keys):
       
        # Create and open output files  
        for file_keys in IO_path_keys:
            self.IO_fileID[file_keys] = open(self.IO_path[file_keys], "w+")


    def __del__(self):
        """ The destructeur defined here is used to ensure that all outputs files are properly closed """ 

        for fileID in self.IO_fileID.values():
            fileID.close()

class Sampling(SolveProblem):


    def __init__(self, input_file_name): 

        SolveProblem.__init__(self, input_file_name)

        new_file_keys = ['MChains', 'MChains_reparam', 'MChains_csv', 'Distribution_values', 'gp', 'aux_variables']
        # Define output file for sampling 
        self.IO_path[new_file_keys[0]] = self.IO_path['out_folder']+"/mcmc_chain.dat"
        self.IO_path[new_file_keys[1]] = self.IO_path['out_folder']+"/mcmc_chain_reparam.dat"
        self.IO_path[new_file_keys[2]] = self.IO_path['out_folder']+"/mcmc_chain.csv"
        self.IO_path[new_file_keys[3]] = self.IO_path['out_folder']+"/distribution_values.csv"
        self.IO_path[new_file_keys[4]] = self.IO_path['out_folder']+"/gp.dat" # Gaussian proposal 
        self.IO_path[new_file_keys[5]] = self.IO_path['out_folder']+"/aux_variables.dat" # momentum variables in HMC and Ito 

        # Create and open files in read-write ('+') mode (w mode erase previous existing files) 
        for file_keys in new_file_keys:
            self.IO_fileID[file_keys] = open(self.IO_path[file_keys], "w+")


    def sample(self, models={}): 

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
        
            # Get hyperparameters 
            n_hyperp = len(self.user_inputs['Sampling']['Distribution']['hyperparameters'])
            n_rand_var = self.user_inputs['Sampling']['Distribution']['n_rand_var']

            for i in range(n_hyperp): 
                if isinstance(self.user_inputs['Sampling']['Distribution']['hyperparameters'][i], str):
                    # Read from file 
                    reader = pd.read_csv(self.user_inputs['Sampling']['Distribution']['hyperparameters'][i], header=None)
                    distr_param.append(reader.values)
                else:
                    distr_param.append(self.user_inputs['Sampling']['Distribution']['hyperparameters'][i])

            if isinstance(self.user_inputs["Sampling"]["Distribution"]["init_val"], str):  
                reader = pd.read_csv(self.user_inputs["Sampling"]["Distribution"]["init_val"], header=None)
                distr_init_val.append(reader.values)
                A = np.transpose(reader.values)
                unpar_init_val = A[0]
            else: 
                distr_init_val.append(self.user_inputs["Sampling"]["Distribution"]["init_val"])
                unpar_init_val = np.array(self.user_inputs["Sampling"]["Distribution"]["init_val"])

            sample_dist = pybitup.distributions.set_probability_dist(distr_name, distr_param, n_rand_var)

            
        elif (self.user_inputs["Sampling"].get("BayesianPosterior") is not None):
            # ----------------------------------------------------------
            # Sample a Bayesian Posterior distribution
            # ----------------------------------------------------------
            # We need to define data and models used for sampling the posterior 

            
            # Get the data sets from file or generate it from the model 
            # ----------------------------------------------------------
            BP_inputs = self.user_inputs["Sampling"]["BayesianPosterior"]
            data = {}
            for data_set, model_id in enumerate(models.keys()):
            
                model_id_reparam = model_id # To be changed. See later in "Posterior computation"

                # Load current data and model properties 
                c_data = BP_inputs['Data'][data_set]
                c_model = BP_inputs['Model'][data_set]
              
                # Model 
                # ------
                # Check if there is an input file for the model 
                if (c_model.get("input_file") is not None): 
                    models[model_id].input_file_name = c_model['input_file'] 

                models[model_id].param = np.array(c_model['param_values'])
                models[model_id].param_nom = np.array(c_model['param_values'])
                models[model_id].param_names = c_model['param_names'] 

                # Data 
                # -----
                if c_data['Type'] == "ReadFromFile": 
                    reader = pd.read_csv(c_data['FileName'])

                    # From the same model (same xField), there can be several yFields for the
                    x = np.array([])
                    y = np.array([])
                    std_y = np.array([])

                    # There is only one xField
                    x = reader[c_data['xField'][0]].values
                    models[model_id].x = x

                    # For a given model_id, there can be several outputs (several yField) for the inference (e.g. output and its derivative(s))
                    # for the same design points x. We put those outputs in a large array of dimension 1 x (n*x)  
                    for nfield, yfield in enumerate(c_data['yField']):     
                        y = np.concatenate((y, reader[yfield].values))
                        std_y = np.concatenate((std_y, reader[c_data['sigmaField'][nfield]].values))
                        dataName = c_data['yField'][0]+"_"+c_data['FileName']

                elif c_data['Type'] == "GenerateSynthetic": 	
                    if  c_data['x']['Type'] == "range": 
                        x = np.array(np.arange(c_data['x']['Value'][0], c_data['x']['Value'][1], c_data['x']['Value'][2]))
                    elif  c_data['x']['Type'] == "linspace": 
                        x = np.array(np.linspace(c_data['x']['Value'][0], c_data['x']['Value'][1], c_data['x']['Value'][2]))

                    models[model_id].x = x 

                    std_y = np.array(c_data['y']['Sigma'])
                    y = np.array(pybitup.bayesian_inference.generate_synthetic_data(models[model_id], c_data['y']['Sigma'], c_data['y']['Error']))
                    dataName = c_data['y']['Name']
                    std_y = np.ones(len(y))*std_y

                else: 
                    raise ValueError("Invalid DataType {}".format(c_data['Type'])) 


                data[model_id] = pybitup.bayesian_inference.Data(dataName, x, y, std_y)

            # Write the data in output data file 
            with open('output/data', 'wb') as file_data_exp: 
                pickle.dump(data, file_data_exp)

            # ----------
            # Inference
            # ----------

            # Get uncertain parameters 
            # -------------------------
            unpar_name = []
            for name in BP_inputs['Prior']['Param'].keys():
                unpar_name.append(name)

            for model_id in models.keys(): 
                models[model_id].unpar_name = unpar_name

            # Get a priori information on the uncertain parameters
            # ----------------------------------------------------
            unpar_init_val = []
            unpar_prior_name = []
            unpar_prior_param = []
            for param_val in BP_inputs['Prior']['Param'].values():
                unpar_init_val.append(param_val['initial_val'])
                unpar_prior_name.append(param_val['prior_name']) 
                unpar_prior_param.append(param_val['prior_param'])
            unpar_init_val = np.array(models[model_id].parametrization_forward(unpar_init_val))
            n_uncertain_param = len(unpar_init_val)

            # Prior 
            # ------ 
            distr_name = [BP_inputs['Prior']['Distribution']]
            hyperparam = [unpar_prior_name, unpar_prior_param]
            prior_dist = pybitup.distributions.set_probability_dist(distr_name, hyperparam, n_uncertain_param)
           
            # Likelihood 
            # -----------
            likelihood_fun = pybitup.bayesian_inference.Likelihood(data, models)

            # ----------------------
            # Posterior computation 
            # ----------------------
            sample_dist = pybitup.bayesian_inference.BayesianPosterior(prior_dist, likelihood_fun, models[model_id_reparam], unpar_init_val) 
            # the model provided here in input is only because it contains the parametrization. They should be the same for all
            # models. Here, we provide the one with model_id = 0 as it is always present. 

        else:
            raise ValueError('No samling distribution provided or invalid name.')

        # Run sampling of the distribution
        if (self.user_inputs["Sampling"].get('Algorithm') is not None): 
            algo = self.user_inputs["Sampling"]["Algorithm"]
            sampling_dist = pybitup.inference_problem.Sampler(self.IO_fileID, sample_dist, algo) 
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
                            

class Propagation(SolveProblem): 

    def propagate(self, models=[]): 

            if (self.user_inputs.get("Propagation") is not None):
                self.input_propagation = self.user_inputs["Propagation"]
            else: 
                raise ValueError('Ask for uncertainty propagation but no Propagation inputs were provided')

            propagation_inputs = self.user_inputs["Propagation"]

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

            if (self.user_inputs["Propagation"]["Model_evaluation"].get("Sample_number") is not None):
                 # Number of sample is provided in the input file 
                n_sample_param = self.user_inputs["Propagation"]["Model_evaluation"]["Sample_number"]
            else:
                # Default number of sample to propagate
                n_sample_param = len(unpar[name])
    


            # Get the design points
            # -----------------------
            # Get the design points (variable input) for each model provided

            n_points = {}
            model_id_to_num = {}
            for model_num, model_id in enumerate(models.keys()): 
                c_model = propagation_inputs["Model"][model_num]

                design_point_input = c_model["design_points"]
                reader = pd.read_csv(design_point_input["filename"])
            
                c_design_points = reader.values[:,design_point_input["field"]]

                model_id_to_num[model_id] = model_num

                # Set the evaluation of the model at the design points 
                models[model_id].x = c_design_points
                n_points[model_id] = models[model_id].size_x()

                # Set the model parameter values 
                # Check if there is an input file for the model 
                if (c_model.get("input_file") is not None): 
                    models[model_id].input_file_name = c_model['input_file'] 

                models[model_id].param = np.array(c_model['param_values'])
                models[model_id].param_nom = np.array(c_model['param_values'])
                models[model_id].param_names = c_model['param_names']
                models[model_id].unpar_name = unpar.keys()


            # Run propagation 
            # ---------------
            # Evaluate the model at the parameter values

            c_param = np.zeros(n_unpar)

            if rank==0:
                
                # Initialize output files 
                output_file_model_eval = {}
                for model_id in models.keys(): 
                    output_file_model_eval[model_id]=open('output/'+model_id+'_eval.csv','ab')

                # Print current time and start clock count
                print("Start time {}" .format(time.asctime(time.localtime())))
                sys.stdout.flush()
                self.t1 = time.clock()


            if (self.user_inputs["Propagation"]["Model_evaluation"].get("Parallel_evaluation") is not None):
                mce = self.user_inputs["Propagation"]["Model_evaluation"]["Parallel_evaluation"]["model_concurrency_evaluation"]
                sce = self.user_inputs["Propagation"]["Model_evaluation"]["Parallel_evaluation"]["sample_concurrency_evaluation"]
                if mce*sce > size: 
                    raise ValueError('Ask for {} model concurrency and {} sample concurrency evaluations but only {} processor(s) were provided'.format(mce, sce, size))
            else: 
                mce = sce = 1

            # Create the list of model evaluation id per rank 
            # The ranks fill first from low to higher number 
            # Initialize 
            model_rank_list = {} 
            for i in range(size):
                 model_rank_list[i] = []


            # Fill the list 
            for model_num, model_id in enumerate(models.keys()): 
                model_rank_list[model_num%mce].append(model_id)

            # If concurrent sample evaluation are asked, we divide
            n_sample_per_proc = int(np.floor(n_sample_param / sce))
            n_sample_per_rank = {}
            rank_sample_list = {} # sample range per rank 

            for i in range(sce):
                for j in range(mce):
                    model_rank_list[i*mce+j] = model_rank_list[j]
                    n_sample_per_rank[i*mce + j] = n_sample_per_proc
                    rank_sample_list[i*mce + j] = [k for k in range(i*n_sample_per_proc, (i+1)*n_sample_per_proc)]
                    if i == (sce-1): 
                        n_sample_per_rank[i*mce + j] = n_sample_per_proc + n_sample_param % sce
                        rank_sample_list[i*mce + j] = [k for k in range(i*n_sample_per_proc, n_sample_param)]

            if rank == 0: 
                for n in model_rank_list.keys(): 
                    print("Proc {} model evaluations: {}; for {} samples in range [{}:{}]".format(n, model_rank_list[n], n_sample_per_rank[n], rank_sample_list[n][0], rank_sample_list[n][-1]))
                    sys.stdout.flush()             

            fun_eval = {}
            data_hist = {}
            for model_id in model_rank_list[rank]: 
                fun_eval[model_id] = np.zeros([n_sample_per_rank[rank], n_points[model_id]])
                data_hist[model_id] = np.zeros([n_sample_per_rank[rank], n_points[model_id]])

            # Iterate over all the parameter values 
            for i, sample_num in enumerate(rank_sample_list[rank]): 
            #for i in range(n_sample_param): 

                if rank==0: 
                    # We estimate time after a hundred iterations
                    if i == 100:
                        print("Estimated time: {}".format(time.strftime("%H:%M:%S",
                                                        time.gmtime((time.clock()-self.t1) / 100.0 * n_sample_per_rank[rank]))))
                        sys.stdout.flush()

                # Get the parameter values 
                for j, name in enumerate(unpar_inputs.keys()):

                    c_param[j] = unpar[name][sample_num]

                # Each proc evaluates the models attributed
                #model_list_per_rank 
                for model_id in model_rank_list[rank]:                 
                    model_num = model_id_to_num[model_id]

                    # Update model evaluation 
                    models[model_id].run_model(c_param)

                    # Get model evaluations
                    fun_eval[model_id][i, :] = models[model_id].model_eval 
                    #fun_eval[model_id][i, :] = self.f_X(c_param, models[model_id], propagation_inputs["Model"][model_num], unpar.keys())

                    c_eval = fun_eval[model_id][i, :]
                    data_hist[model_id][i, :] = c_eval 

            if rank != 0:
                comm.send(fun_eval, dest=0, tag=10)
                comm.send(data_hist, dest=0, tag=11)

            if rank==0: 
                for rank_rcv in range(1, size):   
                    fun_eval_other_proc = comm.recv(source=rank_rcv, tag=10)
                    data_hist_other_proc = comm.recv(source=rank_rcv, tag=11)
                    for model_id in fun_eval_other_proc.keys():     
                        if rank_rcv < mce : 
                            fun_eval[model_id] = fun_eval_other_proc[model_id]
                            data_hist[model_id] = data_hist_other_proc[model_id]
                        else: 
                            fun_eval[model_id] = np.concatenate((fun_eval[model_id], fun_eval_other_proc[model_id]), axis=0)
                            data_hist[model_id] = np.concatenate((data_hist[model_id], data_hist_other_proc[model_id]), axis=0)
                           


                # Rank 0 will compute the statistics 
                data_ij_max = {}
                data_ij_min = {}
                data_ij_mean = {}
                data_ij_var = {}
                for model_id in models.keys(): 
                    data_ij_max[model_id] = -1e5*np.ones(n_points[model_id])
                    data_ij_min[model_id] = 1e5*np.ones(n_points[model_id])
                    data_ij_mean[model_id] = np.zeros(n_points[model_id])
                    data_ij_var[model_id] = np.zeros(n_points[model_id])

                # Iterate over the propagated sample to compute statistics 
                for i in range(n_sample_param):
                    for model_id in models.keys(): 
                        # Write the csv for the function evaluation 
                        np.savetxt(output_file_model_eval[model_id], np.array([fun_eval[model_id][i,:]]), fmt="%f", delimiter=",")

                        c_eval = fun_eval[model_id][i,:]
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
                
                # Save results in output files 
                for model_id in models.keys(): 

                    # Divide by the number of sample to compute the final mean  
                    data_ij_mean[model_id][:] = data_ij_mean[model_id][:]/n_sample_param

                    # Values are saved in csv format using Panda dataframe  
                    df = pd.DataFrame({"mean" : data_ij_mean[model_id][:], 
                                    "lower_bound": data_ij_min[model_id][:], 
                                    "upper_bound": data_ij_max[model_id][:]})
                    df.to_csv('output/'+model_id+"_interval.csv", index=None)

                    df_CI = pd.DataFrame({"CI_lb" : np.percentile(data_hist[model_id], 2.5, axis=0), 
                                        "CI_ub": np.percentile(data_hist[model_id], 97.5, axis=0)})
                    df_CI.to_csv('output/'+model_id+"_CI.csv", index=None) 






        



