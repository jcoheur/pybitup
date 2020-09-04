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
from scipy import stats 

import pybitup.distributions 
import pybitup.bayesian_inference
import pybitup.inference_problem
import pybitup.post_process
import pybitup.solve_problem as sp 


import matplotlib.pyplot as plt

class SensitivityAnalysis(sp.SolveProblem): 


    def __init__(self, input_file_name, model_id): 
        sp.SolveProblem.__init__(self, input_file_name)


        SA_input = self.user_inputs["SensitivityAnalysis"]
        self.model_id = model_id

        # Get uncertain parameters 
        # -------------------------
        # Uncertain parameters are common to all model provided. 

        unpar_inputs = SA_input["Uncertain_param"]
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
                reader = pd.read_csv(c_unpar_input["filename"]) 
                unpar_value = reader.values[:,c_unpar_input["field"]]

                unpar[name] = unpar_value

                n_samples = len(unpar_value)


        # Load function evaluation and sort them in a dictionnary 
        fun_eval = {}
        for j in range(n_samples): 
            fun_eval[j] = np.load("output/{}_fun_eval.{}.npy".format(self.model_id, j)) 

        for i, name in enumerate(unpar_inputs.keys()):

            Expect_tot, Var_param, Var_tot = self.BinObject(unpar, fun_eval, name, bins='auto') 
            #Expect_tot, Var_param, Var_tot = self.BinObject_without_hist(unpar, fun_eval, name, 100) 

            S_i  = Var_param / Var_tot
            #print('S_i', S_i)

            plt.figure(1)
            plt.plot(Expect_tot, 'C0')
            plt.plot(Expect_tot + np.sqrt(Var_param), 'C'+str(i+1), label="S_"+name)
            plt.plot(Expect_tot - np.sqrt(Var_param), 'C'+str(i+1))

            plt.figure(2)
            plt.plot(S_i, 'C'+str(i+1),  label="S_"+name) 

        
        plt.figure(1)
        plt.legend()
        plt.figure(2)
        plt.legend()
        plt.show()

    def BinObject(self, mcmc_chain, fun_eval, param_id, bins=101):
        """ Get the mcmc chain in bins and sort them. Then compute the conditional expectations and variances.
        "mcmc_chain" : full mcmc chains containing the samples from the distribution 
        "fun_eval" is a dictionnary from the ID of the sample (the ID is the index of the sample in the MCMC chain) to the 
        function evaluations. 
        "param_id" is the ID of the parameter in the MCMC chain to be evaluated. 
        "bins" controls the number of bins in which we will sort the sample. "auto" will let the numpy histogram 
        generating them automatically. If a integer n is provided, it will be n equally spaced bins between min 
        and max values of the parameter."""

        mcmc_val_param = mcmc_chain[param_id]
      
        # We can use histogram to define bin edges and the counts in each bin 
        if isinstance(bins, str): 
            print("Bins generated automatically by the numpy.histogram function.")
            r_param, bin_edges = np.histogram(mcmc_val_param, bins='auto')
        else: 
            print("Generate {} bins between min and max param values.".format(bins))
            r_param, bin_edges = np.histogram(mcmc_val_param, range=(np.min(mcmc_val_param), np.max(mcmc_val_param)), bins=bins)
        n_bins = len(r_param)
        # 1. Sort mcmc in increasing order of param ID 
        sample_id_sorted = np.argsort(mcmc_val_param) 

        bin_sample_id_list = list(range(n_bins))
        for i, num_sample in enumerate(r_param): 
            cum_sum = np.sum(r_param[0:i])
            bin_sample_id_list[i]=sample_id_sorted[np.sum(r_param[0:i]):cum_sum+num_sample]


        rj_tot = np.sum(r_param)

        # Compute conditional expectations 
        Expect = []
        Expect_tot = 0
        for i in range(n_bins):
            if r_param[i] == 0: 
                # There is an unpopulated bin 
                print('Bin number {} is unpopulated.'.format(i))
                Expect.append(i)
                Expect[i] = 0
                continue 
            Expect.append(i)
            Expect[i] = 0
            for j in bin_sample_id_list[i]: 
                Expect[i] += fun_eval[j]
            Expect_tot += Expect[i]
            Expect[i] = Expect[i]/r_param[i]
        Expect_tot = Expect_tot / rj_tot

        # Compute conditional varianaces and total variance
        Var_param = 0
        Var_tot = 0
        for i in range(n_bins):
            if r_param[i] == 0: 
                continue 
            Var_param +=  r_param[i] * (Expect[i] - Expect_tot)**2
            for j in bin_sample_id_list[i]: 
                Var_tot += (fun_eval[j] - Expect_tot)**2


        print(rj_tot)
        Var_param = Var_param / rj_tot
        Var_tot = Var_tot / rj_tot

        return Expect_tot, Var_param, Var_tot



    def BinObject_without_hist(self, mcmc_chain, fun_eval, param_id, n_bins):
        """ This is actually the first implementation of BinObject. It is not using the histogram function from numpy. 
        Instead, we implemented something similar with bins of equal size. The IDs of the samples
        sorted in the bins are computed at the same time as the counts.
        See BinObject for explanation of input parameters. """

        mcmc_val_param = mcmc_chain[param_id]

        # Define the bin edges 
        min_param_val = np.min(mcmc_val_param)
        max_param_val = np.max(mcmc_val_param) 
        delta_p_i = (max_param_val- min_param_val)/n_bins 
        bin_edges = [min_param_val]
        for i in range(n_bins-1): 
            bin_edges.append(bin_edges[i] + delta_p_i)
        bin_edges.append(max_param_val)
        

        # 1. Sort mcmc in increasing order of param ID 
        sample_id_sorted = np.argsort(mcmc_val_param) 
        mcmc_parm_val_sorted = mcmc_val_param[sample_id_sorted]

        # 2. Get id number for samples sorted in bins 
        # Initialise 
        num_bin = 0 
        bin_sample_id_list = [num_bin] # Initialise list of sample id sorted in bins 
        r_param =  [num_bin]
        
        bin_sample_id_list[num_bin] = []
        r_param[num_bin] = 0

        for i, sample_val in enumerate(mcmc_parm_val_sorted):
            while sample_val > bin_edges[num_bin+1]: 
                # Update the number of the bin and its upper bound
                num_bin += 1 
                # Create the new list of IDs
                bin_sample_id_list.append(num_bin)
                bin_sample_id_list[num_bin] = []
                r_param.append(num_bin)
                r_param[num_bin] = 0
            
            # Add the id to the list of samples 
            bin_sample_id_list[num_bin].append(sample_id_sorted[i])
            # Add one to the number of elements in the current bin 
            r_param[num_bin] += 1 
        
        rj_tot = np.sum(r_param)
        Expect = []
        Expect_tot = 0
        # 3. Compute conditional expectation 
        for i in range(n_bins):
            #print(i, r_param[i])
            if r_param[i] == 0: 
                # There is an unpopulated bin 
                print('Bin number {} is unpopulated.'.format(i))
                Expect.append(i)
                Expect[i] = 0
                continue 
            Expect.append(i)
            Expect[i] = 0
            for j in bin_sample_id_list[i]: 
                Expect[i] += fun_eval[j]
            Expect_tot += Expect[i]
            Expect[i] = Expect[i]/r_param[i]
        Expect_tot = Expect_tot / rj_tot

        # 4. Compute varianaces 
        Var_param = 0
        Var_tot = 0
        for i in range(n_bins):
            if r_param[i] == 0: 
                continue 
            Var_param +=  r_param[i] * (Expect[i] - Expect_tot)**2
            for j in bin_sample_id_list[i]: 
                Var_tot += (fun_eval[j] - Expect_tot)**2

        print(rj_tot)
        Var_param = Var_param / rj_tot
        Var_tot = Var_tot / rj_tot

        return Expect_tot, Var_param, Var_tot

