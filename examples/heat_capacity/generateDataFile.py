# In this test case, we generate fake data for the inference 
# Fake data are saved in a csv (comma-separated values) file
# This script needs to be run when no data_heat_capacity_0.csv file is present in the case folder. 

# Import modules 
import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 

# Fake data can be generated from the function "generate_synthetic_data" in pybitup.bayesian_inference
from pybitup import bayesian_inference  

# Import the model 
import heat_capacity

# Parameters 
param_model = np.array([-333.33, 4095.83, -2553.33, 570.83])
# Variable
x = np.arange(300.0, 1401.0, 10.0)
# Model definition. Attributes of model are found in pybitup.bayesian_inference
model_def = {}
model_def[0] = heat_capacity.HeatCapacity()
model_def[0].param = param_model
model_def[0].x = x
y_nom = model_def[0].fun_x()

# Define standard deviation on parameters and model 
std_param = param_model / 100
std_y = 50
array_std_y = np.ones(len(x)) * std_y 

# Control the seed so that the same random number are produced each time we run this file 
random.seed(a=0)

# Generate random y 
y_noisy = bayesian_inference.generate_synthetic_data(model_def[0], std_param, array_std_y)

# Plot 
plt.figure(1)
plt.plot(x, y_nom)
plt.plot(x, y_noisy)
plt.legend(["Nominal cp", "Noisy cp"])

plt.show()

# Save data to the .csv file 
df = pd.DataFrame({'T': x, 'cp': y_noisy, 'std_cp': array_std_y})
df.to_csv("data_heat_capacity_0.csv")
		


