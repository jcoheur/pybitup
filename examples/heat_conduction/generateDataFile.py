# The heat conduction test case is from Smith, "Uncertainty quantification: theory, implementation and applications", 2013. 
# In this test case, experimental data are provided. 
# We implement the experimental data here and save them in a csv file. 
# This script needs to be run when no data_heat_conduction_0.csv file is not present in the case folder. 

# Import modules 
import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 

# Import the model 
import heat_conduction

# Nominal parameter  
param = [0.95, 0.95, 2.37, 21.29, -18.41, 0.00191]
x = np.arange(10.0, 70.0, 4.0)
model_def = heat_conduction.HeatConduction()
model_def.param = param
model_def.x = x

# Standard deviation 
std_param = [0, 0, 0, 0, 0.1450, 1.4482e-5] # Not used here 
std_y=0.2504 # 0.2604
array_std_y = np.ones(len(x))
array_std_y *= std_y

# Experimental data provided (see Smith. Tab. 3.2 p. 57, aluminium rod)
y = [96.14, 80.12, 67.66, 57.96, 50.90, 44.84, 39.75, 36.16, 33.31, 31.15, 29.28, 27.88, 27.18, 26.40, 25.86]

df = pd.DataFrame({'x': x, 'T': y, 'std_T': array_std_y})
df.to_csv("heat_conduction_data_0.csv")
		

plt.plot(x, model_def.fun_x())
plt.plot(x, y, 'o', color='C0')

plt.show()
