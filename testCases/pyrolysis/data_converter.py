import numpy as np

# %% Code

point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
np.save("point.npy",point)