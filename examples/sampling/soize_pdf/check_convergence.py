import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import matplotlib.ticker as mtick

name_files = ["mcmc_chain_RWMH_1e5", "mcmc_chain_ISDE_1e5", "mcmc_chain_HMC_1e5"]
n_points = [1,1, 1]
for num_file, my_files in enumerate(name_files): 
    reader = pd.read_csv(my_files+".csv", header=None) 
    param_value_raw = reader.values
    n_samples = len(param_value_raw[:, 0]) + 1


    param_value_raw_ds = param_value_raw[0:n_samples:n_points[num_file], :] 
    n_samples = len(param_value_raw_ds[:, 0]) + 1
    # Computing convergence criteria and graphs for each chains 
    # From Gelman et al., Bayesian Data Analysis, 2014.
    n_unpar = 2 
    for i in range(n_unpar):

        mean_it = np.zeros(n_samples-1)
        plt.figure(1000+i)
        for it in  range(n_samples-1): 
            mean_it[it] = np.mean(param_value_raw_ds[0:it+1, i])

        plt.plot(range(n_samples-1), mean_it)

        plt.legend(["RWMH", "ISDE", "HMC"])
plt.show()
