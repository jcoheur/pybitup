import numpy as np

# %% Code

point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
np.savez_compressed("point.npz",pts=point)