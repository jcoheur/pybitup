
import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 


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

# Experimental data provided (see Smith. Tab. 3.2, aluminium rod)
y = [96.14, 80.12, 67.66, 57.96, 50.90, 44.84, 39.75, 36.16, 33.31, 31.15, 29.28, 27.88, 27.18, 26.40, 25.86]

df = pd.DataFrame({'x': x, 'T': y, 'std_T': array_std_y})
df.to_csv("heat_conduction_data.csv")
		

plt.plot(x, y)
plt.show()
