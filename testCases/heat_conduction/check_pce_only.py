import sys
sys.path.append('../../')
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle

# %% Initialisation

f = open("output/pce_model.pickle","rb")
model = pickle.load(f)
f.close()

f = open("output/pce_poly.pickle","rb")
poly = pickle.load(f)
f.close()

reader = pd.read_csv('test_design_points.dat',header=None)
xMod = reader.values[:]

xMc = np.linspace(0,70,101)
varMc = np.load('var.npy')
meanMc = np.load('mean.npy')
meanMod = model.mean
varMod = model.var

# %% Figures

plt.figure(1)
plt.rcParams.update({"font.size":16})
plt.plot(xMod,meanMod,label="PCE")
plt.plot(xMc,meanMc,'--',label='Monte Carlo')
plt.legend(prop={'size':16})
plt.ylabel("Mean [°C]")
plt.xlabel("$x$ [cm]")
plt.grid()

plt.figure(2)
plt.rcParams.update({"font.size":16})
plt.plot(xMod,varMod,label="PCE")
plt.plot(xMc,varMc,'--',label='Monte Carlo')
plt.legend(prop={'size':16})
plt.ylabel("Variance [°C$^2$]")
plt.xlabel("$x$ [cm]")
plt.grid()

plt.show()