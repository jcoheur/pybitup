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
import math

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

        self.unpar = {} # Dictionnary containing uncertain parameters and their value
        unpar_dist = {}
        for name in unpar_inputs.keys():
            c_unpar_input = unpar_inputs[name]
            if c_unpar_input["filename"] == "None":
                # Get the uncertain parameters from 'labelled' distribution 
                unpar_dist[name] = {} 
                unpar_dist[name]["distribution"]=c_unpar_input['distribution']
                unpar_dist[name]["hyperparameters"]=c_unpar_input['hyperparameters']

                self.unpar[name] = unpar_dist[name] 
                    
            else: # Get the parameter values from the file 
                reader = pd.read_csv(c_unpar_input["filename"]) 
                unpar_value = reader.values[:,c_unpar_input["field"]]

                self.unpar[name] = unpar_value
                self.n_samples = len(unpar_value)
        
        # Load function evaluation and sort them in a dictionnary 
        print("Loading function evaluations for sensitivity analysis ...")
        fun_eval = {}
        for j in range(self.n_samples): 
            fun_eval[j] = np.load("output/{}_fun_eval.{}.npy".format(self.model_id, j)) 



  

        method="0MC" # "2DMC", "MC", "Kernel"
        if method=="MC":
            # Compute conditional variables using Monte Carlo estimation from the MCMC samples 
            for i, name in enumerate(unpar_inputs.keys()):

                Expect_tot, Var_param, Var_tot = self.BinObject(self.unpar, fun_eval, name, bins='auto') 
                #Expect_tot, Var_param, Var_tot = self.BinObject_without_hist(unpar, fun_eval, name, 100) 
                BinObject=BinMethod(self.unpar, name, fun_eval) 
                #BinObject.get_bins(bins='auto') 
                BinObject.get_bins_without_hist(bins=100)
                BinObject.compute_cond_expectation()
                BinObject.compute_variance_cond()
                Var_param = BinObject.Var_param
                Expect_tot = BinObject.Expect_tot
                Var_tot = BinObject.Var_tot

                
                S_i  = Var_param 
                #print('S_i', S_i)

                plt.figure(1)
                plt.plot(Expect_tot, 'C0')
                plt.plot(Expect_tot + np.sqrt(Var_param), color='C'+str(i+1), label="S_"+name)
                plt.plot(Expect_tot - np.sqrt(Var_param), color='C'+str(i+1))

                plt.figure(2)
                plt.plot(S_i,  color='C'+str(i+1),  label="S_"+name) 
                plt.plot(Var_tot, color="black", label="V_tot") 
            
            plt.figure(1)
            plt.legend()
            plt.figure(2)
            plt.legend()
 
        method="2DMC"
        if method=="2DMC":
            # Compute conditional variables using Monte Carlo estimation from the MCMC samples 
            name = ["A", "E"] # for i, name in enumerate(unpar_inputs.keys()):
            i = 4


            #Expect_tot, Var_param, Var_tot = self.BinObject_hist2d(self.unpar, fun_eval, name, bins='auto') 
            #Expect_tot, Var_param, Var_tot = self.BinObject_without_hist(unpar, fun_eval, name, 100) 

            BinObject2D=BinMethod2D(self.unpar, name, fun_eval) 
            BinObject2D.get_bins(bins='auto') 
            BinObject2D.compute_cond_expectation()
            BinObject2D.compute_variance_cond()
            Var_param = BinObject2D.Var_param
            Expect_tot = BinObject2D.Expect_tot
            Var_tot = BinObject2D.Var_tot

            S_i  = Var_param 
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


        data = {}
        method="0Kernel2d" # "MC", "Kernel"
        names=[["A", "E"], ["A", "n"], ["E", "n"], ["A", "F"], ["E", "F"], ["n", "F"]]
        if method=="Kernel2d":
        # Compute conditional expectation and variance using Kernel Method 
            for i, name in enumerate(names):
                #KernelObject=KernelMethod2D(self.unpar, name, fun_eval) 
                KernelObject=KernelMethodDD(self.unpar, name, fun_eval) 
                KernelObject.compute_variance_cond()
                V_i = KernelObject.V_i

                S_i = V_i
                data_name = "S_"+name[0]+name[1]
                data[data_name] = V_i
                
                plt.figure(3)
                plt.plot(S_i, '--', color='C'+str(i+1),  label=data_name) 

        method="0Kernel3d" # "MC", "Kernel"
        names=[["E", "n", "F"]]
        if method=="Kernel3d":
        # Compute conditional expectation and variance using Kernel Method 
            for i, name in enumerate(names):
                #KernelObject=KernelMethod3D(self.unpar, name, fun_eval) 
                KernelObject=KernelMethodDD(self.unpar, name, fun_eval) 
                KernelObject.compute_variance_cond()
                V_i = KernelObject.V_i

                S_i = V_i
                data_name = "S_"+name[0]+name[1]+name[2]
                data[data_name] = V_i
                
                plt.figure(3)
                plt.plot(S_i, '--', color='C'+str(i+1),  label=data_name)    


        method="0KernelDD" # "MC", "Kernel"
        names=[["A", "E", "n", "F"]]
        if method=="KernelDD":
        # Compute conditional expectation and variance using Kernel Method 
            for i, name in enumerate(names):
                KernelObject=KernelMethodDD(self.unpar, name, fun_eval) 
                KernelObject.compute_variance_cond()
                V_i = KernelObject.V_i

                S_i = V_i
                data_name = "S_"+name[0]+name[1]+name[2]+name[3]
                data[data_name] = V_i
                
                plt.figure(3)
                plt.plot(S_i, '--', color='C'+str(i+1),  label=data_name)   

        method="0Kernel" # "MC", "Kernel"
        if method=="Kernel":
        # Compute conditional expectation and variance using Kernel Method 
            for i, name in enumerate(unpar_inputs.keys()):

                print("Computing sensitivity index for {}.".format(name))
                KernelObject=KernelMethod(self.unpar, name, fun_eval) 
                #KernelObject=KernelMethodDD(self.unpar, name, fun_eval) 

                KernelObject.compute_variance_cond()
                #KernelObject.compute_variance_cond_without_hist()

                V_i = KernelObject.V_i

                S_i = V_i

                data_name = "S_"+name
                data[data_name] = V_i
                plt.figure(3)
                plt.plot(S_i, 'C'+str(i+1),  label=data_name) 
                


                #KernelObject.compute_cond_expectation()
        # data["V_tot"] = Var_tot
        # df = pd.DataFrame(data)
        # df.to_csv("sensitivity_values.csv", header=True)
        plt.figure(3)
        plt.plot(Var_tot, color="black", label="V_tot") 
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
        print(r_param)
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

    def BinObject_hist2d(self, mcmc_chain, fun_eval, param_id, bins=101):
        """ Get the mcmc chain in bins and sort them. Then compute the conditional expectations and variances.
        "mcmc_chain" : full mcmc chains containing the samples from the distribution 
        "fun_eval" is a dictionnary from the ID of the sample (the ID is the index of the sample in the MCMC chain) to the 
        function evaluations. 
        "param_id" is the ID of the parameter in the MCMC chain to be evaluated. 
        "bins" controls the number of bins in which we will sort the sample. "auto" will let the numpy histogram 
        generating them automatically. If a integer n is provided, it will be n equally spaced bins between min 
        and max values of the parameter."""

        mcmc_val_param_x = mcmc_chain[param_id[0]]
        mcmc_val_param_y = mcmc_chain[param_id[1]]      

        # We can use histogram to define bin edges and the counts in each bin 
        if isinstance(bins, str): 
            print("Bins generated automatically by the numpy.histogram function.")
            r_param, x_edges, y_edges = np.histogram2d(mcmc_val_param_x, mcmc_val_param_y, bins=(21,21))


        np_array_fun_eval = np.array(fun_eval)
        transpose_f = np_array_fun_eval.T
        list_fun_eval_bis = []
        for val in fun_eval:
            list_fun_eval_bis.append(fun_eval[val].tolist())
        np_array_fun_eval_bis= np.array(list_fun_eval_bis)
        fun_eval_bis_transpose = np_array_fun_eval_bis.T
        list_fun_eval_bis_transpose = fun_eval_bis_transpose.tolist()


        ret=stats.binned_statistic_2d(mcmc_val_param_x, mcmc_val_param_y, list_fun_eval_bis_transpose, bins=10, expand_binnumbers=True)
        print(len(ret.binnumber[0]))

        n_bins = len(r_param)
        # 1. Sort mcmc in increasing order of param ID 
        sample_id_sorted_x = np.argsort(mcmc_val_param_x) 
        sample_id_sorted_y = np.argsort(mcmc_val_param_y) 

        bin_sample_id_list = list(range(n_bins))
        for i, num_sample in enumerate(r_param): 
            cum_sum = np.sum(r_param[0:i])
            bin_sample_id_list[i]=sample_id_sorted_x[np.sum(r_param[0:i]):cum_sum+num_sample]


        rj_tot = np.sum(r_param)
        Expect = []
        Expect_tot = 0
        # Compute conditional expectations 
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


class BinMethod():
        """ Get the mcmc chain in bins and sort them. Then compute the conditional expectations and variances.
        "mcmc_chain" : full mcmc chains containing the samples from the distribution 
        "fun_eval" is a dictionnary from the ID of the sample (the ID is the index of the sample in the MCMC chain) to the 
        function evaluations. 
        "param_id" is the ID of the parameter in the MCMC chain to be evaluated. 
        "bins" controls the number of bins in which we will sort the sample. "auto" will let the numpy histogram 
        generating them automatically. If a integer n is provided, it will be n equally spaced bins between min 
        and max values of the parameter."""

        def __init__(self, unpar, param_name, fun_eval): 

            self.mcmc_val_param = unpar[param_name]
            self.fun_eval = fun_eval 


        def get_bins_without_hist(self, bins=101): 
            """ Get the bin and bin edges without the use of numpy histogram function. """

            # Define the bin edges 
            min_param_val = np.min(self.mcmc_val_param)
            max_param_val = np.max(self.mcmc_val_param) 
            delta_p_i = (max_param_val- min_param_val)/bins 
            bin_edges = [min_param_val]
            for i in range(bins-1): 
                bin_edges.append(bin_edges[i] + delta_p_i)
            bin_edges.append(max_param_val)
            

            # 1. Sort mcmc in increasing order of param ID 
            sample_id_sorted = np.argsort(self.mcmc_val_param) 
            mcmc_parm_val_sorted = self.mcmc_val_param[sample_id_sorted]

            # 2. Get id number for samples sorted in bins 
            # Initialise 
            num_bin = 0 
            self.bin_sample_id_list = [num_bin] # Initialise list of sample id sorted in bins 
            self.r_param =  [num_bin]
            
            self.bin_sample_id_list[num_bin] = []
            self.r_param[num_bin] = 0

            for i, sample_val in enumerate(mcmc_parm_val_sorted):
                while sample_val > bin_edges[num_bin+1]: 
                    # Update the number of the bin and its upper bound
                    num_bin += 1 
                    # Create the new list of IDs
                    self.bin_sample_id_list.append(num_bin)
                    self.bin_sample_id_list[num_bin] = []
                    self.r_param.append(num_bin)
                    self.r_param[num_bin] = 0
                
                # Add the id to the list of samples 
                self.bin_sample_id_list[num_bin].append(sample_id_sorted[i])
                # Add one to the number of elements in the current bin 
                self.r_param[num_bin] += 1 

            self.n_bins = len(self.r_param)
            self.rj_tot = np.sum(self.r_param)

        def get_bins(self, bins=101):   
            """ Get the bin and bin edges. 
            We can use numpy histogram functions to define bin edges and the counts in each bin."""      

            if isinstance(bins, str): 
                print("Bins generated automatically by the numpy.histogram function.")
                self.r_param, self.bin_edges = np.histogram(self.mcmc_val_param, bins='auto')
            else: 
                print("Generate {} bins between min and max param values.".format(bins))
                self.r_param, self.bin_edges = np.histogram(self.mcmc_val_param, range=(np.min(self.mcmc_val_param), np.max(self.mcmc_val_param)), bins=bins)
            print(self.r_param)
            self.n_bins = len(self.r_param)
            # 1. Sort mcmc in increasing order of param ID 
            sample_id_sorted = np.argsort(self.mcmc_val_param) 

            self.bin_sample_id_list = list(range(self.n_bins))
            for i, num_sample in enumerate(self.r_param): 
                cum_sum = np.sum(self.r_param[0:i])
                self.bin_sample_id_list[i]=sample_id_sorted[np.sum(self.r_param[0:i]):cum_sum+num_sample]

            self.rj_tot = np.sum(self.r_param)

        def compute_cond_expectation(self):
            # Compute conditional expectations 
            self.Expect = []
            self.Expect_tot = 0
            for i in range(self.n_bins):
                if self.r_param[i] == 0: 
                    # There is an unpopulated bin 
                    print('Bin number {} is unpopulated.'.format(i))
                    self.Expect.append(i)
                    self.Expect[i] = 0
                    continue 
                self.Expect.append(i)
                self.Expect[i] = 0
                for j in self.bin_sample_id_list[i]: 
                    self.Expect[i] += self.fun_eval[j]
                self.Expect_tot += self.Expect[i]
                self.Expect[i] = self.Expect[i]/self.r_param[i]
            self.Expect_tot = self.Expect_tot / self.rj_tot

        def compute_variance_cond(self):

            # Compute conditional varianaces and total variance
            self.Var_param = 0
            self.Var_tot = 0
            for i in range(self.n_bins):
                if self.r_param[i] == 0: 
                    continue 
                self.Var_param +=  self.r_param[i] * (self.Expect[i] - self.Expect_tot)**2
                for j in self.bin_sample_id_list[i]: 
                    self.Var_tot += (self.fun_eval[j] - self.Expect_tot)**2


            print(self.rj_tot)
            self.Var_param = self.Var_param / self.rj_tot
            self.Var_tot = self.Var_tot / self.rj_tot


class BinMethod2D(BinMethod):
    """ Get the mcmc chain in bins and sort them. Then compute the conditional expectations and variances.
    "mcmc_chain" : full mcmc chains containing the samples from the distribution 
    "fun_eval" is a dictionnary from the ID of the sample (the ID is the index of the sample in the MCMC chain) to the 
    function evaluations. 
    "param_id" is the ID of the parameter in the MCMC chain to be evaluated. 
    "bins" controls the number of bins in which we will sort the sample. "auto" will let the numpy histogram 
    generating them automatically. If a integer n is provided, it will be n equally spaced bins between min 
    and max values of the parameter."""

    def __init__(self, unpar, param_name, fun_eval): 
        BinMethod.__init__(self, unpar, param_name, fun_eval)

        self.mcmc_val_param_x = unpar[param_name[0]]
        self.mcmc_val_param_y = unpar[param_name[1]]   

    def get_bins_without_hist(self, bins=101): 
        raise ValueError("Function get_bins_without_hist not available in two dimensions")

    def get_bins(self, bins=101): 
        # We use the numpy histogram2d function to define bin edges and the counts in each bin 
        if isinstance(bins, str): 
            print("Bins generated automatically by the numpy.histogram function.")
            self.r_param, x_edges, y_edges = np.histogram2d(self.mcmc_val_param_x, self.mcmc_val_param_y, bins=(21,21))


        np_array_fun_eval = np.array(self.fun_eval)
        transpose_f = np_array_fun_eval.T
        list_fun_eval_bis = []
        for val in self.fun_eval:
            list_fun_eval_bis.append(self.fun_eval[val].tolist())
        np_array_fun_eval_bis= np.array(list_fun_eval_bis)
        fun_eval_bis_transpose = np_array_fun_eval_bis.T
        list_fun_eval_bis_transpose = fun_eval_bis_transpose.tolist()


        ret=stats.binned_statistic_2d(self.mcmc_val_param_x, self.mcmc_val_param_y, list_fun_eval_bis_transpose, bins=10, expand_binnumbers=True)
        print(len(ret.binnumber[0]))

        self.n_bins = len(self.r_param)
        # 1. Sort mcmc in increasing order of param ID 
        sample_id_sorted_x = np.argsort(self.mcmc_val_param_x) 
        #sample_id_sorted_y = np.argsort(self.mcmc_val_param_y) # Not used  

        self.bin_sample_id_list = list(range(self.n_bins))
        for i, num_sample in enumerate(self.r_param): 
            cum_sum = np.sum(self.r_param[0:i])
            self.bin_sample_id_list[i]=sample_id_sorted_x[np.sum(self.r_param[0:i]):cum_sum+num_sample]

        self.rj_tot = np.sum(self.r_param)

class KernelMethod(): 


    def __init__(self, unpar, param_name, fun_eval): 


        self.mcmc_val_param = unpar[param_name]
        self.fun_eval = fun_eval 
        self.n_samples = len(self.mcmc_val_param)
        self.sigma_x = np.sqrt(np.var(self.mcmc_val_param))
        self.h_gauss = 1.06*self.sigma_x*self.n_samples**(-1/5)
        self.a_exp = 1/np.sqrt(2*np.pi)/self.h_gauss


    def compute_cond_expectation(self, x):

        sum_q_Gk = 0 
        sum_Gk = 0
        for j, x_j in enumerate(self.mcmc_val_param): 
            Gk = self.gaussKernel(x-x_j, self.h_gauss)
            q_j = self.fun_eval[j]

            sum_q_Gk += q_j*Gk
            sum_Gk += Gk

        f_u = sum_q_Gk/sum_Gk

        return f_u


    def compute_variance_cond(self):

        sum_cond_expect = 0
        sum_cond_expect_2 = 0
        r_param, bin_edges = np.histogram(self.mcmc_val_param, bins='auto')

        for i, num_sample in enumerate(r_param): 
            get_sum = self.compute_cond_expectation(bin_edges[i+1])
            sum_cond_expect += get_sum**2*num_sample/self.n_samples 
            sum_cond_expect_2 += get_sum*num_sample/self.n_samples 

        self.V_i = sum_cond_expect  - sum_cond_expect_2**2 

    def compute_variance_cond_without_hist(self):

        sum_cond_expect = 0
        sum_cond_expect_2 = 0

        for i, x_i in enumerate(self.mcmc_val_param[0::]): 
            get_sum = self.compute_cond_expectation(x_i)
            sum_cond_expect += get_sum**2/self.n_samples
            sum_cond_expect_2 += get_sum/self.n_samples 

        self.V_i = sum_cond_expect  - sum_cond_expect_2**2 

    def gaussKernel(self, x, h): 
        """ Compute the Gaussian Kernel. 
        Inputs: 
            - x: evaluation points.
            - h: bandwith of the kernel.
        Output: 
            - y     
        Initial Matlab implementation from Fevrier and Gregov."""

        y = self.a_exp*np.exp(-0.5*((x/h)**2))

        return y





class KernelMethod2D(): 


    def __init__(self, unpar, param_name, fun_eval): 


        self.mcmc_val_param_x = unpar[param_name[0]]
        self.mcmc_val_param_y = unpar[param_name[1]]      

        self.fun_eval = fun_eval 
        self.n_samples = len(self.mcmc_val_param_x)
        self.sigma_x = np.sqrt(np.var(self.mcmc_val_param_x))
        self.sigma_y = np.sqrt(np.var(self.mcmc_val_param_y))
        self.h_gauss_x = 1.06*self.sigma_x*self.n_samples**(-1/5)
        self.h_gauss_y = 1.06*self.sigma_y*self.n_samples**(-1/5)
        self.a_exp = 1/(2*np.pi)/(self.h_gauss_x * self.h_gauss_y)



    def compute_cond_expectation(self, x):

        sum_q_Gk = 0 
        sum_Gk = 0
        for j, val_x in enumerate(self.mcmc_val_param_x): 
            x_j = [val_x, self.mcmc_val_param_y[j]]
            Gk = self.gaussKernel(x-x_j, [self.h_gauss_x, self.h_gauss_y])
            q_j = self.fun_eval[j]

            sum_q_Gk += q_j*Gk
            sum_Gk += Gk

        if sum_Gk == 0: 
            f_u = 0 
        else: 
            f_u = sum_q_Gk/sum_Gk

        return f_u


    def compute_variance_cond(self):

        sum_cond_expect = 0
        sum_cond_expect_2 = 0

        r_param, x_edges, y_edges = np.histogram2d(self.mcmc_val_param_x, self.mcmc_val_param_y, bins=(21,21))

        for i, n_param_i in enumerate(r_param):
            for j, n_param_ij in enumerate(n_param_i): 
                get_sum = self.compute_cond_expectation(np.array([x_edges[i+1],y_edges[j+1]]))
                sum_cond_expect += get_sum**2*n_param_ij/self.n_samples 
                sum_cond_expect_2 += get_sum*n_param_ij/self.n_samples 

        self.V_i = sum_cond_expect  - sum_cond_expect_2**2 

    def gaussKernel(self, x, h): 
        """ Compute the Gaussian Kernel. 
        Inputs: 
            - x: evaluation points.
            - h: bandwith of the kernel.
        Output: 
            - y     
        Initial Matlab implementation from Fevrier and Gregov."""
        y = self.a_exp*np.exp(-0.5*((x[0]/h[0])**2 + (x[1]/h[1])**2))

        return y



class KernelMethod3D(): 


    def __init__(self, unpar, param_name, fun_eval): 


        self.mcmc_val_param_x = unpar[param_name[0]]
        self.mcmc_val_param_y = unpar[param_name[1]]      
        self.mcmc_val_param_z = unpar[param_name[2]]    

        self.fun_eval = fun_eval 
        self.n_samples = len(self.mcmc_val_param_x)
        self.sigma_x = np.sqrt(np.var(self.mcmc_val_param_x))
        self.sigma_y = np.sqrt(np.var(self.mcmc_val_param_y))
        self.sigma_z = np.sqrt(np.var(self.mcmc_val_param_z))
        self.h_gauss_x = 1.06*self.sigma_x*self.n_samples**(-1/5)
        self.h_gauss_y = 1.06*self.sigma_y*self.n_samples**(-1/5)
        self.h_gauss_z = 1.06*self.sigma_z*self.n_samples**(-1/5)
        self.a_exp = 1/(2*np.pi)/(self.h_gauss_x * self.h_gauss_y * self.h_gauss_z)



    def compute_cond_expectation(self, x):

        sum_q_Gk = 0 
        sum_Gk = 0
        for j, val_x in enumerate(self.mcmc_val_param_x): 
            x_j = [val_x, self.mcmc_val_param_y[j], self.mcmc_val_param_z[j]]
            array_h_auss = [self.h_gauss_x, self.h_gauss_y, self.h_gauss_z]
            Gk = self.gaussKernel(x-x_j, array_h_auss)
            q_j = self.fun_eval[j]

            sum_q_Gk += q_j*Gk
            sum_Gk += Gk

        if sum_Gk == 0: 
            f_u = 0 
        else: 
            f_u = sum_q_Gk/sum_Gk

        return f_u


    def compute_variance_cond(self):

        sum_cond_expect = 0
        sum_cond_expect_2 = 0

        r_param, edges = np.histogramdd([self.mcmc_val_param_x, self.mcmc_val_param_y, self.mcmc_val_param_z], bins=(21,21,21))

        for i, n_param_i in enumerate(r_param):
            for j, n_param_ij in enumerate(n_param_i):   
                for k, n_param_ijk in enumerate(n_param_ij): 
                    get_sum = self.compute_cond_expectation(np.array([edges[0][i+1], edges[1][j+1],edges[2][k+1]]))
                    sum_cond_expect += get_sum**2*n_param_ijk/self.n_samples 
                    sum_cond_expect_2 += get_sum*n_param_ijk/self.n_samples 

        self.V_i = sum_cond_expect  - sum_cond_expect_2**2 

    def gaussKernel(self, x, h): 
        """ Compute the Gaussian Kernel. 
        Inputs: 
            - x: evaluation points.
            - h: bandwith of the kernel.
        Output: 
            - y     
        Initial Matlab implementation from Fevrier and Gregov."""
        y = self.a_exp*np.exp(-0.5*((x[0]/h[0])**2 + (x[1]/h[1])**2 + (x[2]/h[2])**2))

        return y


class KernelMethodDD(): 


    def __init__(self, unpar, param_name, fun_eval): 


        self.n_samples = len(unpar[param_name[0]])
        self.param_name = param_name

        self.mcmc_val_param = []
        self.sigma = []
        self.h_gauss = []
        self.a_exp = 1
        for i, name in enumerate(param_name): 
            self.mcmc_val_param.append(i)
            self.sigma.append(i)
            self.h_gauss.append(i)

            self.mcmc_val_param[i] = unpar[name]
            self.sigma[i] = np.sqrt(np.var(self.mcmc_val_param[i]))
            self.h_gauss[i] = 1.06*self.sigma[i]*self.n_samples**(-1/5) # Optimal for Gaussian random variable
            self.a_exp /= self.h_gauss[i] * np.sqrt(2*np.pi)
            

        self.fun_eval = fun_eval 
        self.n_dim = len(param_name)

    def compute_cond_expectation(self, x):

        sum_q_Gk = 0 
        sum_Gk = 0
        
        for j in range(self.n_samples):
            x_j = np.zeros(self.n_dim)
            array_h_gauss = np.zeros(self.n_dim)
            for i in range(self.n_dim): 
                x_j[i] = self.mcmc_val_param[i][j]
                array_h_gauss[i] = self.h_gauss[i] # could be done in __init__

            Gk = self.gaussKernel(x-x_j, array_h_gauss)
            q_j = self.fun_eval[j]

            sum_q_Gk += q_j*Gk
            sum_Gk += Gk

        if sum_Gk == 0: 
            f_u = 0 
        else: 
            f_u = sum_q_Gk/sum_Gk

        return f_u

    def compute_variance_cond_2D(self):
        """ For 3D, it might be faster and more accurate to use histogram2d function than C integration."""

        sum_cond_expect = 0
        sum_cond_expect_2 = 0

        r_param, x_edges, y_edges = np.histogram2d(self.mcmc_val_param[0], self.mcmc_val_param[1], bins=(21,21))

        for i, n_param_i in enumerate(r_param):
            for j, n_param_ij in enumerate(n_param_i): 
                get_sum = self.compute_cond_expectation(np.array([x_edges[i+1],y_edges[j+1]]))
                sum_cond_expect += get_sum**2*n_param_ij/self.n_samples 
                sum_cond_expect_2 += get_sum*n_param_ij/self.n_samples 

        self.V_i = sum_cond_expect  - sum_cond_expect_2**2 

    def compute_variance_cond_3D(self):
        """ For 3D, it might be faster and more accurate to use histogramdd function."""

        sum_cond_expect = 0
        sum_cond_expect_2 = 0

        r_param, edges = np.histogramdd([self.mcmc_val_param[0], self.mcmc_val_param[1], self.mcmc_val_param[2]], bins=(21,21,21))

        for i, n_param_i in enumerate(r_param):
            for j, n_param_ij in enumerate(n_param_i):   
                for k, n_param_ijk in enumerate(n_param_ij): 
                    get_sum = self.compute_cond_expectation(np.array([edges[0][i+1], edges[1][j+1],edges[2][k+1]]))
                    sum_cond_expect += get_sum**2*n_param_ijk/self.n_samples 
                    sum_cond_expect_2 += get_sum*n_param_ijk/self.n_samples 

        self.V_i = sum_cond_expect  - sum_cond_expect_2**2 

    # For higher dimensions, I couldn't generalized to high dimenson using histogrammdd because of the multiple loops. 
    # def compute_variance_con_DD(self):

    #     sum_cond_expect = 0
    #     sum_cond_expect_2 = 0

    #     param_array = []
    #     for i, name in enumerate(self.param_name): 
    #         param_array.append(self.mcmc_val_param[name])
    #         bin_tuple += (10)

    #     r_param, edges = np.histogramdd(param_array, bins=bin_tuple)

    #     # How to compute this intrication of loop for a general number of loop ????
    #     for i, n_param_i in enumerate(r_param):
    #         for j, n_param_ij in enumerate(n_param_i):   
    #             for k, n_param_ijk in enumerate(n_param_ij): 
    #                 get_sum = self.compute_cond_expectation(np.array([edges[0][i+1], edges[1][j+1],edges[2][k+1]]))
    #                 sum_cond_expect += get_sum**2*n_param_ijk/self.n_samples 
    #                 sum_cond_expect_2 += get_sum*n_param_ijk/self.n_samples 

    #     self.V_i = sum_cond_expect  - sum_cond_expect_2**2 

    def compute_variance_cond(self):
        """ Compute conditional variance by performing Monte Carlo integration."""

        sum_cond_expect = 0
        sum_cond_expect_2 = 0

        n_MC = 200 # Number of Monte Carlo samples we want from the Markov chain 
        delta_val = math.floor(self.n_samples/n_MC)    
        for j in range(0,self.n_samples,delta_val): 
            param_array_j = np.zeros(self.n_dim)
            for i in range(self.n_dim): 
                param_array_j[i]  = self.mcmc_val_param[i][j]

            get_sum = self.compute_cond_expectation(param_array_j)
            sum_cond_expect += get_sum**2/n_MC
            sum_cond_expect_2 += get_sum/n_MC

        self.V_i = sum_cond_expect  - sum_cond_expect_2**2 


    def gaussKernel(self, x, h): 
        """ Compute the Gaussian Kernel. 
        Inputs: 
            - x: evaluation points.
            - h: bandwith of the kernel.
        Output: 
            - y     
        Initial Matlab implementation from Fevrier and Gregov."""

        arg_sum = 0
        for i in range(self.n_dim): 
            arg_sum += (x[i]/h[i])**2

        y = self.a_exp*np.exp(-0.5*arg_sum)

        return y
