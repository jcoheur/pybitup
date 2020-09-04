# Pyrolysis test case (J. Coheur) 
# In this test case, we generate fake experimental data from the model and save them in a csv file. 
# This script needs to be run when no data_pyrolysis_0.csv file is not present in the case folder. 

# Import modules 
import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 

# Fake data can be generated from the function "generate_synthetic_data" in pybitup.bayesian_inference
from pybitup import bayesian_inference  

# Import the model 
import one_reaction_pyrolysis

# Nominal parameter  
param_nom = np.array([1.6635e4, 113000, 2.0, 0.0,  0.04, 6.1])
x = np.linspace(300.0, 1400.0, 201)

# Model definition. Attributes of model are found in pybitup.bayesian_inference
model_def = {}
model_def[0] = one_reaction_pyrolysis.OneReactionPyrolysis()
model_def[0].param = param_nom
model_def[0].x = x
y_nom = model_def[0].fun_x()

# Define standard deviation on parameters and data 
std_param = param_nom / 100  
std_y = 0.00001 
array_std_y = np.ones(len(x)) * std_y 

# Control the seed so that the same random number are produced each time we run this file 
random.seed(a=0)

# Generate random y 
y_noisy = bayesian_inference.generate_synthetic_data(model_def[0], std_param, array_std_y)

# Plot nominal data and noisy data
plt.figure(1)
plt.plot(x, y_nom)
plt.plot(x, y_noisy)
plt.legend(["Nominal prod", "Noisy prod"])
plt.show()

# Save data to the .csv file 
df = pd.DataFrame({'T': x, 'gas_prod': y_noisy, 'std_gas_prod': array_std_y})
df.to_csv("data_pyrolysis_0.csv")
