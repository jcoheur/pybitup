from scipy import linalg
import random 
import numpy as np 
import pandas as pd 

# test_rosenthal

# Generate the random covariance to be sampled 
# This example is taken from "G. Roberts and J. Rosenthal, Examples of Adaptive MCMC, Journal of Computational and Graphical Statistics, 2009."
d = 100

M = np.zeros([d,d])
mean = np.zeros([d])
for i in range(0, d): 
    for j in range(0, d):
        M[i][j] = random.gauss(0, 1)

gauss_cov = M * M.T + 100 * np.eye(d)
print(gauss_cov)
df = pd.DataFrame(mean)
df.to_csv("mean.csv", header=None, index=None)
df.to_csv("init_val.csv", header=None, index=None)
df_prop_cov = pd.DataFrame(np.ones([d])*0.2)
df_prop_cov.to_csv("prop_cov.csv", header=None, index=None)

df2 = pd.DataFrame({'col_0': gauss_cov[0][:]})
for i in range(1, d): 
    df2['col_'+str(i)] = gauss_cov[i][:]

# Check that determinant of covariance mtric is positive 
det_cov = linalg.det(gauss_cov)
print(det_cov)
print(linalg.det(linalg.inv(gauss_cov)))

df2.to_csv("cov.csv", header=None, index=None)





