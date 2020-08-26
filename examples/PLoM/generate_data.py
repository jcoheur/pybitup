import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
import random 

# Titan modules 
import create_model 

## Generate initial data set 

N = 400
sig3 = 1 # 3-sigma error (controls the statistical fluctuation) 
a = 10 # radius of the helix
b = 33/(2*np.pi)   # 2*pi*b step of the helix
epsihelical = 1 # -1 or 1

theta = np.zeros((N, 1)) #Initialise column vector 
for i in range(0, N): 
    theta[i] = 100/b*random.random()

X1 = a*np.cos(theta)
X2 = a*epsihelical*np.sin(theta)
X3 = b*theta

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1,X2,X3, c='b', marker='o', s=0.5)
ax.set_xlim(-20,20)
ax.set_ylim(-20,20)
ax.set_zlim(-10,100)


# We add the statistical fluctuation 
X1_ = np.zeros((N, 1)) #Initialise column vector 
X2_ = np.zeros((N, 1)) #Initialise column vector 
X3_ = np.zeros((N, 1)) #Initialise column vector 
for i in range(0, N): 
    X1_[i] = X1[i] + sig3/3*random.gauss(0, 1)
    X2_[i] = X2[i] + sig3/3*random.gauss(0, 1)
    X3_[i] = X3[i] + sig3/3*random.gauss(0, 1)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1_,X2_,X3_, c='b', marker='o', s=0.5)
ax.set_xlim(-20,20)
ax.set_ylim(-20,20)
ax.set_zlim(-10,100)



## 3 - Create probabilistic representation of the data

PLoM_model = create_model.TitanPLoM() 
PLoM_model.Type = 'PLoM'
PLoM_model.ExpDesign.X = np.concatenate((X1_, X2_), axis=1) #X must be column vectors 
PLoM_model.ExpDesign.Y = X3_

PLoM_model.Opt.scaling = 0
PLoM_model.Opt.optimizationm = 0
PLoM_model.Opt.epsvalue = 1.57     # value of the smoothing parameter (is not determined with the optimization procedure)
PLoM_model.Opt.m = 4

PLoM_model.titan_PLoM() 

plt.show()


# 4 - Sample new realizations from the probabilistic representation of the data

PLoM_model.Itoopt.nMC = 10
PLoM_model.Itoopt.M0 = 110
PLoM_model.Itoopt.l0 = 0
PLoM_model.Itoopt.dt = 0.1196
Y = titan_PLoM_eval(PLoM_model)

# # 5 - Processing of new realizations

# nMC = Metamodel.Itoopt.nMC
# X1new = zeros(N*nMC,1)
# X2new = zeros(N*nMC,1)
# X3new = zeros(N*nMC,1) 

# for ll=1:nMC
#     for ii=1:N
#         X1new((ll-1)*N+ii) = Y(1,ii,ll)
#         X2new((ll-1)*N+ii) = Y(2,ii,ll)
#         X3new((ll-1)*N+ii) = Y(3,ii,ll)
#     end
# end

# figure
# scatter3(X1,X2,X3,'blue')
# hold on
# scatter3(X1new,X2new,X3new,'red')
# xlim([-20,20])
# ylim([-20,20])
# zlim([-10,100])

# figure
# plot(X1new)

# figure
# plot(X2new)

# figure
# plot(X3new)


