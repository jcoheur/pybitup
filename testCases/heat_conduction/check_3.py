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

point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
index = []
resp = []

for i in range(len(point)):
    try:
        resp.append(np.load("output/heat_conduction_3_fun_eval."+str(i)+".npy"))
        index.append(i)
    except: pass

resp = np.array(resp)
respMod = model.eval(point[index])

# %% Monte Carlo and error

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)
meanMod = np.mean(respMod,axis=0)
varMod = np.var(respMod,axis=0)

x = [10,14,18,22,26,30,34,38,42,46,50,54,58,62,66]
xMod = np.linspace(0,70,71)

# %% Figures

plt.figure(1)
plt.rcParams.update({"font.size":16})
plt.plot(xMod,meanMod,'C0',label="PCE")
plt.plot(x,mean,'C1--',label="MC")
plt.legend(prop={'size':16})
plt.ylabel("Mean")
plt.xlabel("x")
plt.grid()

plt.figure(2)
plt.rcParams.update({"font.size":16})
plt.plot(xMod,varMod,'C0',label="PCE")
plt.plot(x,var,'C1--',label="MC")
plt.legend(prop={'size':16})
plt.ylabel("Variance")
plt.xlabel("x")
plt.grid()

plt.figure(3)
plt.rcParams.update({"font.size":16})
plt.plot(point[:,0],point[:,1],".C0")
plt.xlabel("$O$")
plt.ylabel("$h$")
plt.grid()

plt.show()