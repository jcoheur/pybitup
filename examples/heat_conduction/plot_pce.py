import sys
sys.path.append('../../')
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd 
# %% Initialisation

model_name = "heat_conduction" 

f = open("output/propagation/pce_model_"+model_name+".pickle","rb")
model = pickle.load(f)
f.close()

f = open("output/propagation/pce_poly_"+model_name+".pickle","rb")
poly = pickle.load(f)
f.close()

point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
index = []
resp = []

for i in range(len(point)):
    try:
        resp.append(np.load("output/heat_conduction_fun_eval."+str(i)+".npy"))
        index.append(i)
    except: pass

resp = np.array(resp)
respMod = model.eval(point[index])

# %% Monte Carlo and error

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)
meanMod = np.mean(respMod,axis=0)
varMod = np.var(respMod,axis=0)

xMC = np.arange(10.0,70.0,4)
reader = pd.read_csv('test_design_points.csv')
xMod = reader["T"].values


# %% Figures

plt.figure(1)
plt.rcParams.update({"font.size":16})
plt.plot(xMod,meanMod,'C0',label="PCE")
plt.plot(xMC,mean,'C1--',label="MC")
plt.legend(prop={'size':16})
plt.ylabel("Mean")
plt.xlabel("x")
plt.grid()

plt.figure(2)
plt.rcParams.update({"font.size":16})
plt.plot(xMod,varMod,'C0',label="PCE")
plt.plot(xMC,var,'C1--',label="MC")
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