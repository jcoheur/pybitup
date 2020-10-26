# The spring model test case is from Smith, "Uncertainty quantification: theory, implementation and applications", 2013. 
# We implement the "fake" experimental data here and save them in a csv file. 
# This script needs to be run when no spring_model_data_0.csv file is not present in the case folder. 

# Import modules 
import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 

# Import the model 
import spring_model

# Fake data can be generated from the function "generate_synthetic_data" in pybitup.bayesian_inference
from pybitup import bayesian_inference 

# Nominal parameter  
param = [1.5, 20.5]
x = np.linspace(0.0, 5.0, 51)

model_def = {}
model_def[0] = spring_model.SpringModel()
model_def[0].param = param
model_def[0].x = x
y_nom = model_def[0].fun_x()

# Define standard deviation on parameters and data  
std_param = [0.0, 0.0] # param / 100 
std_y = .1
array_std_y = np.ones(len(x)) * std_y 

# Control the seed so that the same random number are produced each time we run this file 
random.seed(a=0)

# Generate random y 
y_noisy = bayesian_inference.generate_synthetic_data(model_def[0], std_param, array_std_y)

std_y = 0.3
array_std_y = np.ones(len(x)) * std_y 

# Plot 
plt.figure(1)
plt.plot(x, y_nom)
plt.plot(x, y_noisy, 'o', color='C0')
plt.legend(["Displacement (nominal)", "Displacement (noisy)"])

df = pd.DataFrame({'time': x, 'd': y_noisy, 'std_d': array_std_y})
df.to_csv("spring_model_data_0.csv")
		
plt.show()
