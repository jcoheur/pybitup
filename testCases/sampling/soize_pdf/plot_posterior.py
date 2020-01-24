import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
# Plot results for the posterior 

vec_param_i = np.linspace(-1.5, 1.5, 100)
delta_param_i = vec_param_i[1] - vec_param_i[0]
vec_param_j = np.linspace(-1.5, 2, 100)
delta_param_j = vec_param_j[1] - vec_param_j[0]


f_post = np.load("2d_post.npy")

marginal_post_1 = np.sum(f_post*delta_param_j, axis=1)
int_f_post  = np.sum(marginal_post_1*delta_param_i, axis=0)
norm_f_post = f_post / int_f_post

marginal_post_norm_1 = np.sum(norm_f_post*delta_param_j, axis=1)
marginal_post_norm_2 = np.sum(norm_f_post*delta_param_i, axis=0)
plt.figure(200)
plt.plot(vec_param_i, marginal_post_norm_1)
plt.figure(201)
plt.plot(vec_param_j, marginal_post_norm_2)



# #  2D MCMC iterations no rescaling 
# --------------------------------
plt.figure(203)
plt.contour(vec_param_i, vec_param_j, f_post.T)

reader = pd.read_csv('output/mcmc_chain.csv') 
param_value_raw = reader.values
n_samples = len(param_value_raw[:, 0]) + 1

n_points = 1
plt.plot(param_value_raw[0:n_samples:n_points, 0], param_value_raw[0:n_samples:n_points, 1], '.', markersize=1,  alpha=0.8)
plt.plot(param_value_raw[0, 0], param_value_raw[0, 1], 'r*', )
plt.xlabel("x_1")
plt.ylabel("x_2")









plt.show()

