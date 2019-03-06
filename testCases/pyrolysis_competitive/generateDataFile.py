import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 


# Add to the current data file of Bessire and Wong a column with FAKE standard deviation 

# filename = 'Bessire_366Kmin'
filename = 'Wong_10Kmin'

df = pd.read_csv(filename+'.csv')


	
std_y=0.005
df['std_dRho'] = str(std_y)

df.to_csv(filename+"_withFakeStd.csv")
	

