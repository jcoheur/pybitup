import numpy as np

# %% Code

pts = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
point = np.zeros((pts.shape[0],8))

point[:,0] = pts[:,3]
point[:,1] = pts[:,0]
point[:,2] = pts[:,1]
point[:,3] = pts[:,2]
point[:,4] = pts[:,7]
point[:,5] = pts[:,4]
point[:,6] = pts[:,5]
point[:,7] = pts[:,6]

np.savez_compressed("point.npz",pts=point)