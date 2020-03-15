import sys
sys.path.append('../../')
import numpy as np
from matplotlib import pyplot as plt
import pickle

# %% Initialisation

T = np.linspace(300,1400,101)
f = open("output/pce_model.pickle","rb")
model = pickle.load(f)
f.close()

f = open("output/pce_poly.pickle","rb")
poly = pickle.load(f)
f.close()

point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
index = []
resp = []

for i in range(len(point)):
    try:
        resp.append(np.load("output/one_reaction_pyrolysis_fun_eval."+str(i)+".npy"))
        index.append(i)
    except: pass

resp = np.array(resp)
respMod = model.eval(point[index])
respMod = respMod

# %% Monte Carlo and error

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)
meanMod = np.mean(respMod,axis=0)
varMod = np.var(respMod,axis=0)

error = abs(np.divide(resp-respMod,resp))
error = 100*np.mean(error,axis=0)

# %% Figures

plt.figure(1)
plt.rcParams.update({"font.size":16})
plt.plot(meanMod,'C0',label="PCE")
plt.plot(mean,'C1--',label="MC")
plt.legend(prop={'size':16})
plt.ylabel("Response")
plt.xlabel("Step")
plt.grid()

plt.figure(2)
plt.rcParams.update({"font.size":16})
plt.plot(varMod,'C0',label="PCE")
plt.plot(var,'C1--',label="MC")
plt.legend(prop={'size':16})
plt.ylabel("Response")
plt.xlabel("Step")
plt.grid()