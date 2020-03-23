import numpy as np

# %% Code

point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
nbrPts = len(point)
resp = np.array([np.load("output/pyrolysis_parallel_fun_eval."+str(i)+".npy") for i in range(0,nbrPts)])
np.save("resp.npy",resp)
np.save("point.npy",point)