import numpy as np

# %% Code

point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
nbrPts = len(point)
try: resp = np.array([np.load("output/pyrolysis_parallel_fun_eval."+str(i)+".npy") for i in range(0,nbrPts)])
except: resp = np.array([np.load("output/pyrolysis_parallel_rescaled_fun_eval."+str(i)+".npy") for i in range(0,nbrPts)])

np.savez_compressed("point.npz",pts=point)
np.savez_compressed("resp.npz",resp=resp)