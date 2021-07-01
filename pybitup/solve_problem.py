import os
import json
from jsmin import jsmin
import pathlib 
import shutil 

import pandas as pd
import pickle
import time
import sys

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

        # Initialise path object and file IDs dict
        self.IO_util = {}
        self.IO_util['path'] = {}
        self.IO_util['fileID'] = {} 

        # Define paths and files 
        p_cwd = pathlib.Path.cwd() 
        print(f"The current working directory is {p_cwd}.")
        self.IO_util['path']['cwd'] = p_cwd
        self.IO_util['path']['out_folder'] = pathlib.Path(p_cwd, "output")

        # Create the output folder if it does not exist
        if self.IO_util['path']['out_folder'].exists():
            print(f"{self.IO_util['path']['out_folder']} already exists.")
        else: 
            print(f"Creating output directory {self.IO_util['path']['out_folder']}")
            self.IO_util['path']['out_folder'].mkdir()

        # Path to model evaluations folder (for Bayesian inference, SA 
        # and propagation):
        self.IO_util['path']['fun_eval_folder'] = pathlib.Path(self.IO_util['path']['out_folder'], "model_eval") 

        # Output text file where we write info regarding the case 
        self.IO_util['path']['out_data'] = pathlib.Path(self.IO_util['path']['out_folder'], "output.txt")

        # 'a+': append mode with creation if does not exists  
        self.IO_util['fileID']['out_data'] = open(self.IO_util['path']['out_data'], 'a+')

        # Copy input file in the output folder to keep track of it 
        print(f"Copying {input_file_name} file into output folder ... ")
        shutil.copy(input_file_name, self.IO_util['path']['out_folder'])

    def __del__(self):
        """ The destructeur defined here is used to ensure that all outputs files are properly closed """ 

        for fileID in self.IO_util['fileID'].values(): 
            fileID.close()


class Sampling(SolveProblem):


    def __init__(self, input_file_name): 

        SolveProblem.__init__(self, input_file_name)
        new_file_list = {'MChains': "mcmc_chain.csv", 
                         'MChains_reparam': "mcmc_chain_reparam.csv"}

        # Other outputs for sampling 
        # TODO: maybe it would be better to build this list when we read the input files, and generate all fileID after reading all input file? 
        if (self.user_inputs["Sampling"].get('Algorithm') is not None): 
            self.algo_inputs = self.user_inputs["Sampling"].get('Algorithm')
            # Estimation of sigma (optional)
            if self.algo_inputs.get("estimate_sigma") is not None:
                if self.algo_inputs['estimate_sigma'] == "yes": 
                    new_file_list['estimated_sigma'] = "estimated_sigma.csv"

            # Auxiliary variables for ISDE and HMC 
            if (self.algo_inputs.get("ISDE") is not None) or (self.algo_inputs.get("HMC") is not None): 
                new_file_list['aux_variables'] = "aux_variables.csv"

            # Estimation of maximum distribution (MAP for Bayesian inference)
            if self.algo_inputs.get("estimate_max_distr") is not None: 
                if self.algo_inputs['estimate_max_distr'] == "yes": 
                    new_file_list['estimate_arg_max_val_distr'] = "arg_MAP_estimation.csv"
                    new_file_list['estimate_max_val_distr'] = "log_MAP_estimation.csv"

        # Others optional output? 

        # Define output file for sampling 
        for key in new_file_list.keys():
            self.IO_util['path'][str(key)] = pathlib.Path(self.IO_util['path']['out_folder'], new_file_list[key]) 
            # If we do not use 'str(key), the Pathlib object is not a file. 
            # print(self.IO_util['path'][str(key)].is_file())

        # Create and open files in read-write ('+') mode (w mode erase previous existing files) 
        for key in new_file_list.keys():
            self.IO_util['fileID'][str(key)] = open(self.IO_util['path'][str(key)], "w+")

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

            # For Bayesian posterior, model evaluations will be saved in a 
            # folder. Create the output folder if it does not exist
            if self.IO_util['path']['fun_eval_folder'].exists():
                print(f"{self.IO_util['path']['fun_eval_folder']} already exists.")
            else: 
                print(f"Creating output directory {self.IO_util['path']['fun_eval_folder']}")
                self.IO_util['path']['fun_eval_folder'].mkdir()

            
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

                    # Get the number of experimental runs 
                    if (c_data.get("n_runs") is not None): 
                        # If we specify the number of runs, data files must be named accordigly 
                        # e.g. my_filename_0.csv, my_filename_1.csv, ..., my_filename_nruns.csv.  
                        n_runs = c_data['n_runs']
                        app_name = "_0.csv"
                    else: 
                        # If not specified, default is one and no suffix at the end of the filename
                        n_runs = 1
                        app_name = ".csv"

                    # Read input csv file 
                    data_name = c_data['FileName']
                    data_filename = data_name+app_name
                    reader = pd.read_csv(data_filename)

                    # Initialise arrays 
                    # From the same model (same xField), there can be several yFields 
                    x = np.array([])
                    y = np.array([])
                    std_y = np.array([])

                    # There is only one xField
                    x = reader[c_data['xField'][0]].values
                    models[model_id].x = x 

                    y_tot = {}
                    for c_run in range(n_runs): 
                        # For a given model_id, there can be several outputs (several yField) for the inference (e.g. output and its derivative(s))
                        # for the same design points x. We put those outputs in a large array of dimension 1 x (n*x) 
                        y = [] 
                        std_y = []

                        data_filename = data_name+app_name
                        reader = pd.read_csv(data_filename)

                        for nfield, yfield in enumerate(c_data['yField']):     
                            y = np.concatenate((y, reader[yfield].values))
                            std_y = np.concatenate((std_y, reader[c_data['sigmaField'][nfield]].values))
                            dataName = c_data['yField'][0]+"_"+data_name
                        y_tot[c_run] = y

                        # If there are several input experimental runs, then we iterate on app_name to read them 
                        app_name = "_"+str(c_run)+'.csv'


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


                data[model_id] = pybitup.bayesian_inference.Data(dataName, x, y_tot, std_y)



            # Create data file in the output folder and write data in it with
            # pickle
            self.IO_util['path']['data'] = pathlib.Path(self.IO_util['path']['out_folder'], "data.bin")
            with open(self.IO_util['path']['data'], 'wb') as file_data_exp: 
                pickle.dump(data, file_data_exp)



            # ----------
            # Inference
            # ----------

            # Get uncertain parameters 
            # -------------------------
            unpar_name = []
            unpar_name_dict = {}
            for i, name in enumerate(BP_inputs['Prior']['Param'].keys()):
                unpar_name.append(name)
                unpar_name_dict[name] = i

            for model_id in models.keys(): 
                models[model_id].unpar_name = unpar_name
                models[model_id].unpar_name_dict = unpar_name_dict

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

            # Unpar init val from file if there is a mcmc_chain_init.csv file 
            init_mcm_file_path = pathlib.Path(self.IO_util['path']['cwd'], "mcmc_chain_init.csv")
            if init_mcm_file_path.exists():
                reader = pd.read_csv(init_mcm_file_path, header=None)
                param_value_raw = reader.values
                unpar_init_val = np.array(models[model_id].parametrization_forward(np.float64(param_value_raw[0, :])))

            # Prior 
            # ------ 
            distr_name = [BP_inputs['Prior']['Distribution']]
            hyperparam = [unpar_prior_name, unpar_prior_param]
            prior_dist = pybitup.distributions.set_probability_dist(distr_name, hyperparam, n_uncertain_param)
           
            # Likelihood 
            # -----------
            if (BP_inputs['Likelihood'].get('gamma') is not None): 
                gamma = BP_inputs['Likelihood']['gamma']
            else: 
                gamma = 1.0    
            likelihood_fun = pybitup.bayesian_inference.Likelihood(data, models, gamma)

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
            sampling_dist = pybitup.inference_problem.Sampler(self.IO_util, sample_dist, algo) 
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

    def __init__(self, input_file_name): 
        SolveProblem.__init__(self, input_file_name)

        self.IO_util['path']['out_folder_prop'] = pathlib.Path(self.IO_util['path']['out_folder'], "propagation")

        # Create the output folder if it does not exist
        if self.IO_util['path']['out_folder_prop'].exists():
            print(f"{self.IO_util['path']['out_folder_prop']} already exists.")
        else: 
            print(f"Creating output directory {self.IO_util['path']['out_folder_prop']}")
            self.IO_util['path']['out_folder_prop'].mkdir()


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
            unpar_dist = {}
            for name in unpar_inputs.keys():
                c_unpar_input = unpar_inputs[name]
                if c_unpar_input["filename"] == "None":
                    # Get the uncertain parameters from 'labelled' distribution 
                    unpar_dist[name] = {} 
                    unpar_dist[name]["distribution"]=c_unpar_input['distribution']
                    unpar_dist[name]["hyperparameters"]=c_unpar_input['hyperparameters']

                    unpar[name] = unpar_dist[name] 
                       
                else: # Get the parameter values from the file 
                    if type(c_unpar_input["field"]) is str: 
                        # The field is a str and we read it 
                        reader = pd.read_csv(c_unpar_input["filename"]) 
                        unpar_value = reader[c_unpar_input["field"]].values[:]
                    else: 
                        # We provide a int number so there is no header in the 
                        # mcmc file
                        reader = pd.read_csv(c_unpar_input["filename"], header=None) 
                        unpar_value = reader[c_unpar_input["field"]].values[:]

        
                    unpar[name] = unpar_value


            # Get the design points
            # -----------------------
            # Get the design points (variable input) for each model provided

            n_points = {}
            model_id_to_num = {}
            for model_num, model_id in enumerate(models.keys()): 
                c_model = propagation_inputs["Model"][model_num]

                design_point_input = c_model["design_points"]
                reader = pd.read_csv(design_point_input["filename"])
            
                c_design_points = reader[design_point_input["field"]].values # reader.values[:,design_point_input["field"]]

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

            # Create surrogate
            # ----------------
            # Polynomial Chaos Method

            for model_id in models.keys():

                if self.user_inputs["Propagation"]["Model"][0]["emulator"]=="pce":  # [0] need to be change if we have several models ? 

                    print("Computing pce of", model_id)

                    # Set pce parameters
                    pce_param = self.user_inputs["Propagation"]["Model"][0]["pce"]
                    pce = pybitup.polynomial_chaos.PCE(pce_param, unpar_dist)

                    # Compute pce
                    poly,coef,model = pce.compute_pce(models[model_id])

                    # Save the pce model in output
                    pce_model_path_file = pathlib.Path(self.IO_util['path']
                    ['out_folder_prop'], f"pce_model_{model_id}.bin") 
                    pce_poly_path_file = pathlib.Path(self.IO_util['path']
                    ['out_folder_prop'], f"pce_poly_{model_id}.bin") 
                    pce.save_pickle(model, pce_model_path_file)
                    pce.save_pickle(poly, pce_poly_path_file)


            if self.user_inputs["Propagation"]["Model"][0]["emulator"]=="None": 

                if (self.user_inputs["Propagation"]["Model_evaluation"].get("Sample_number") is not None):
                    # Number of sample is provided in the input file 
                    n_sample_param = self.user_inputs["Propagation"]["Model_evaluation"]["Sample_number"]

                    for name in unpar_inputs.keys():
                        delta_sample = int(len(unpar[name][:])/(n_sample_param-1))
                        # for delta_sample, we divide by n-1 in order to have n 
                        # intervals of  equal length
                        unpar[name] = unpar[name][::delta_sample]

                    
                else:
                    # Default number of sample to propagate
                    n_sample_param = len(unpar[name])
                    print(n_sample_param)

                # Run propagation
                # ---------------
                # Evaluate the model at the parameter values

                c_param = np.zeros(n_unpar)

                if rank==0:
                    
                    # Initialize output files 
                    output_file_model_eval = {}
                    for model_id in models.keys(): 
                        init_model_eval_file = pathlib.Path(self.IO_util['path']['out_folder_prop'], f"{model_id}_fun_evals.npy") 
                        output_file_model_eval[model_id] = open(init_model_eval_file,'ab')

                    # Print current time and start clock count
                    print(f"Start time {time.asctime(time.localtime())}", flush=True)
                    self.t1 = time.perf_counter()

                if (self.user_inputs["Propagation"]["Model_evaluation"].get("Parallel_evaluation") is not None):
                    mce = self.user_inputs["Propagation"]["Model_evaluation"]["Parallel_evaluation"]["model_concurrency_evaluation"]
                    sce = self.user_inputs["Propagation"]["Model_evaluation"]["Parallel_evaluation"]["sample_concurrency_evaluation"]
                    if mce*sce > size: 
                        raise ValueError(f"Ask for {mce} model concurrency and {sce} sample concurrency evaluations but only {size} processor(s) were provided")
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
                        print(f"Proc {n} model evaluations: {model_rank_list[n]}; for {n_sample_per_rank[n]} samples in range [{rank_sample_list[n][0]}:{rank_sample_list[n][-1]}]", flush = True)

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
                        # TODO: do this earlier than 100 .. 
                        if i == 100:
                            print("Estimated time: {}".format(time.strftime("%H:%M:%S",
                                                            time.gmtime((time.perf_counter()-self.t1) / 100.0 * n_sample_per_rank[rank]))))
                            sys.stdout.flush()

                    # Get the parameter values 
                    for j, name in enumerate(unpar_inputs.keys()):

                        c_param[j] = unpar[name][sample_num]

                    # Each proc evaluates the models attributed
                    # model_list_per_rank 
                    for model_id in model_rank_list[rank]:                 
                        model_num = model_id_to_num[model_id]

                        # HERE SHOULD BE UPDATED IF EMULATOR WAS ASKED
                        # Update model evaluation 
                        # print(f"Proc {rank}: param {c_param}")
                        models[model_id].run_model(c_param)
                        # response = model.eval(point)
                        
                        

                        # Get model evaluations
                        fun_eval[model_id][i, :] = models[model_id].model_eval 
                        #fun_eval[model_id][i, :] = self.f_X(c_param, models[model_id], propagation_inputs["Model"][model_num], unpar.keys())
                    

                        c_eval = fun_eval[model_id][i, :]
                        data_hist[model_id][i, :] = c_eval 

                # Other procs send fun_eval and data_hist 
                if rank != 0:
                    comm.send(fun_eval, dest=0, tag=10)
                    comm.send(data_hist, dest=0, tag=11)
                # proc 0 receive data 
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
                    # TODO: put this in the init 
                    self.IO_util['path']['out_folder_model_eval_prop'] = pathlib.Path(self.IO_util['path']['out_folder_prop'], "model_eval")
                    if self.IO_util['path']['out_folder_model_eval_prop'].exists():
                        pass
                    else: 
                        self.IO_util['path']['out_folder_model_eval_prop'].mkdir()
                    # File extension
                    me_suffix = "npy" #"csv"
                    for i in range(n_sample_param):
                        for model_id in models.keys(): 
                            # Write the csv for the function evaluation 
                            np.savetxt(output_file_model_eval[model_id], np.array([fun_eval[model_id][i,:]]), fmt="%f", delimiter=",")
                            
                            path_to_me = pathlib.Path(self.IO_util['path']['out_folder_model_eval_prop'], f"{model_id}_fun_eval-{i}.{me_suffix}")
                            if me_suffix == "npy":
                                np.save(path_to_me, fun_eval[model_id][i,:])
                            else:
                                df = pd.DataFrame({"y": fun_eval[model_id][i,:]})
                                df.to_csv(path_to_me, index=False, header=None) 

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
                        "%H:%M:%S", time.gmtime(time.perf_counter()-self.t1))))
                    
                    # Save results in output files 
                    for model_id in models.keys(): 

                        # Divide by the number of sample to compute the final mean  
                        data_ij_mean[model_id][:] = data_ij_mean[model_id][:]/n_sample_param

                        # Values are saved in csv format using Panda dataframe  
                        df = pd.DataFrame({"x" : models[model_id].x,
                                        "mean" : data_ij_mean[model_id][:], 
                                        "lower_bound": data_ij_min[model_id][:], 
                                        "upper_bound": data_ij_max[model_id][:]})

                        path_to_interv_file = pathlib.Path(self.IO_util['path']['out_folder_prop'], f"{model_id}_interval.csv") 
                        df.to_csv(path_to_interv_file, index=None)

                        df_CI = pd.DataFrame({"x" : models[model_id].x,
                                            "CI_lb" : np.percentile(data_hist[model_id], 2.5, axis=0), 
                                            "CI_ub": np.percentile(data_hist[model_id], 97.5, axis=0)})
                        path_to_CI_file = pathlib.Path(self.IO_util['path']['out_folder_prop'], f"{model_id}_CI.csv") 
                        df_CI.to_csv(path_to_CI_file, index=None) 






        



