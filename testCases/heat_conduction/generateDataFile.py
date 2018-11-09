
import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 


import heat_conduction
# Create fake data in a tsv (tab-separated values) file 


param = [0.95, 0.95, 2.37, 21.29, -18.41, 0.00191]
x = np.array([np.arange(10.0, 70.0, 4.0)])
y = heat_conduction.model_def(x, param)
std_y=1.0
array_std_y = np.ones(len(y))
array_std_y *= std_y


num_data = len(x[0,:])
rn_data=np.zeros((1, num_data))
for i in range(0, num_data):
	rn_data[0,i]=random.gauss(0, std_y)

y += rn_data[0,:]

df = pd.DataFrame({'x': x[0,:], 'T': y, 'std_T': array_std_y})
df.to_csv("heat_conduction_data.csv")
		

plt.plot(x[0,:], y)
plt.show()
