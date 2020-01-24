from scipy import stats
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

#gauss_cov = M * M.T + 100 * np.eye(d)
gauss_cov = np.matmul(M.T, M) 

df = pd.DataFrame(mean)
df.to_csv("mean.csv", header=None, index=None)
df.to_csv("init_val.csv", header=None, index=None)
df_prop_cov = pd.DataFrame(np.ones([d])*0.002)
df_prop_cov.to_csv("prop_cov.csv", header=None, index=None)

df2 = pd.DataFrame({'col_0': gauss_cov[0][:]})
for i in range(1, d): 
    df2['col_'+str(i)] = gauss_cov[i][:]

# Check that determinant of covariance mtric is positive 
det_cov = linalg.det(gauss_cov)
print(det_cov)
print(linalg.det(linalg.inv(gauss_cov)))

df2.to_csv("cov.csv", header=None, index=None)



## Generate the random matrix from Soize 

# Parameters 
delta = 0.5

norm_sup_delta = (d + 1)**(1/2) * (d + 5)**(-1/2)
#print("norm sup delta = {} ".format(norm_sup_delta))


sigma_L = delta * (d + 1)**(-1/2)
# gamma parameters 
a = (d + 1)/(2 * delta**2) + (1-j)/2

L = np.zeros([d,d])
for i in range(0, d): 
    for j in range(i, d):
        if i == j: 
            valgamma = stats.gamma.rvs(a)
            L[i][j] = sigma_L * np.sqrt(2 * valgamma) 
        else: 
            L[i][j] = sigma_L * random.gauss(0, 1)

G = np.matmul(L.T, L) 

df_matG = pd.DataFrame({'col_0': G[0][:]})
for i in range(1, d): 
    df_matG['col_'+str(i)] = G[i][:]

df_matG.to_csv("mat_G.csv", header=None, index=None)

# Check that determinant of covariance mtric is positive 
det_cov = linalg.det(G)
print(det_cov)
print(linalg.det(linalg.inv(G)))




