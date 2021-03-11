import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pybitup.post_process import saveToTikz

# Plot results for the posterior 


n_cuvres_contourf = 5

# Analytical posterior 
#--------------------------

vec_param_i = np.linspace(-4, 4, 200)
delta_param_i = vec_param_i[1] - vec_param_i[0]
vec_param_j = np.linspace(-4, 4, 200)
delta_param_j = vec_param_j[1] - vec_param_j[0]


f_post = np.load("2d_post_bivariate_gauss.npy")

marginal_post_1 = np.sum(f_post*delta_param_j, axis=1)
int_f_post  = np.sum(marginal_post_1*delta_param_i, axis=0)
norm_f_post = f_post / int_f_post

# Plot marginal pdfs 
marginal_post_norm_1 = np.sum(norm_f_post*delta_param_j, axis=1)
marginal_post_norm_2 = np.sum(norm_f_post*delta_param_i, axis=0)

plt.figure(200)
plt.plot(vec_param_i, marginal_post_norm_1)

plt.figure(201)
plt.plot(vec_param_j, marginal_post_norm_2)

# Plot 2d pdf
plt.figure(202)
plt.contour(vec_param_i, vec_param_j, f_post.T, n_cuvres_contourf)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])


# 2D MCMC RWMH
# ------------------

plt.figure(203)
plt.contour(vec_param_i, vec_param_j, f_post.T, n_cuvres_contourf)
reader = pd.read_csv('mcmc_chain_rwmh_400Iterations_2DGauss.csv', header=None) 
param_value_raw = reader.values
n_samples = len(param_value_raw[:, 0]) + 1

n_points = 20
plt.plot(param_value_raw[0:n_samples:n_points, 0], param_value_raw[0:n_samples:n_points, 1], '-o', markersize=4,  alpha=0.8)
plt.plot(param_value_raw[0, 0], param_value_raw[0, 1], 'r*', )

plt.xlabel("x_1")
plt.ylabel("x_2")
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])

saveToTikz('2d_gauss_rmwh.tex')


# Hamiltonian Monte Carlo 
# -----------------------

plt.figure(204)
plt.contour(vec_param_i, vec_param_j, f_post.T, n_cuvres_contourf)
# mcmc_chain_HMC_40Iterations_2DGauss.csv
reader = pd.read_csv('output/mcmc_chain.csv', header=None) 
param_value_raw = reader.values
n_samples = len(param_value_raw[:, 0]) + 1

n_points = 1
plt.plot(param_value_raw[0:n_samples:n_points, 0], param_value_raw[0:n_samples:n_points, 1], '-o', markersize=4,  alpha=0.8)
plt.plot(param_value_raw[0, 0], param_value_raw[0, 1], 'r*', )

plt.xlabel("x_1")
plt.ylabel("x_2")
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])

# saveToTikz('2d_gauss_hmc.tex')

# Ito-SDE Markov Chain Monte Carlo 
# --------------------------------

plt.figure(205)
plt.contour(vec_param_i, vec_param_j, f_post.T, n_cuvres_contourf)
reader = pd.read_csv('mcmc_chain_Ito_400Iterations_2DGauss.csv', header=None) 
param_value_raw = reader.values
n_samples = len(param_value_raw[:, 0]) + 1

n_points = 20
plt.plot(param_value_raw[0:n_samples:n_points, 0], param_value_raw[0:n_samples:n_points, 1], '-o', markersize=4,  alpha=0.8)
plt.plot(param_value_raw[0, 0], param_value_raw[0, 1], 'r*', )

plt.xlabel("x_1")
plt.ylabel("x_2")
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])


plt.show()









