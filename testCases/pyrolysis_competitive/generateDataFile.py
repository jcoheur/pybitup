import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 


# Add to the current data file of Bessire and Wong a column with FAKE standard deviation 

df = pd.read_csv('Bessire_366Kmin.csv') 


	
std_y=0.01	
df['std_dRho'] = str(std_y)

df.to_csv("Bessire_366Kmin_withFakeStd.csv")
	

