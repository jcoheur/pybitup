
import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 
import pybitup 

import heat_conduction

# Create fake data in a tsv (tab-separated values) file 
param = [0.95, 0.95, 2.37, 21.29, -18.41, 0.00191]
x = np.arange(10.0, 70.0, 4.0)
model_def = heat_conduction.HeatConduction()
model_def.param = param
model_def.x = x


std_y=0.2604
array_std_y = np.ones(len(x))
array_std_y *= std_y

# Generate experimental data from deterministic simulation and random error from std
#y = model_def.compute_temperature()
#num_data = len(x)
#rn_data=np.zeros((1, num_data))
#for i in range(0, num_data):
#	rn_data[0,i]=random.gauss(0, std_y)
#y += rn_data[0,:]

# Experimental data provided (see Smith. Tab. 3.2 p. 57, aluminium rod)
y = [96.14, 80.12, 67.66, 57.96, 50.90, 44.84, 39.75, 36.16, 33.31, 31.15, 29.28, 27.88, 27.18, 26.40, 25.86]

df = pd.DataFrame({'x': x, 'T': y, 'std_T': array_std_y})
df.to_csv("heat_conduction_data.csv")
		

plt.plot(x, y)
plt.show()


## Generate the MCMC chains and estimate mean and variance of the response 

case_name = "heat_conduction_2"
input_file_name = "{}.json".format(case_name) 

heat_conduction_model = {}
heat_conduction_model[case_name] = heat_conduction.HeatConduction()

post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(heat_conduction_model)
post_dist.__del__()

point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
index = []
resp = []

for i in range(len(point)):
    try:
        resp.append(np.load("output/heat_conduction_2_fun_eval."+str(i)+".npy"))
        index.append(i)
    except: pass


resp = np.array(resp)

# %% Monte Carlo and error

varMC = np.var(resp,axis=0)
meanMC = np.mean(resp,axis=0)

np.save('mean_from_mcmc.npy',meanMC)
np.save('var_from_mcmc.npy',varMC)
