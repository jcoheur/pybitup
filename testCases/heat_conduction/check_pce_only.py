import sys
sys.path.append('../../')
import numpy as np
from matplotlib import pyplot as plt
import pickle

# %% Initialisation

f = open("output/pce_model.pickle","rb")
model = pickle.load(f)
f.close()

f = open("output/pce_poly.pickle","rb")
poly = pickle.load(f)
f.close()

meanMod = model.mean
varMod = model.var

# %% Figures

plt.figure(1)
plt.rcParams.update({"font.size":16})
plt.plot(meanMod,'C0',label="PCE")
plt.legend(prop={'size':16})
plt.ylabel("Mean")
plt.xlabel("Step")
plt.grid()

plt.figure(2)
plt.rcParams.update({"font.size":16})
plt.plot(varMod,'C0',label="PCE")
plt.legend(prop={'size':16})
plt.ylabel("Variance")
plt.xlabel("Step")
plt.grid()


plt.show()