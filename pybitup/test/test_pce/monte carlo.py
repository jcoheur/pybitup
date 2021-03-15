from matplotlib import pyplot as plt
import numpy as np

# %% Functions

def fun(x,O,h):

    a = 0.95
    b = 0.95
    L = 70
    k = 2.37
    Tamb = 21.29

    gam = np.sqrt((2*(a+b)*h)/(a*b*k))
    C1 = -O/(k*gam)*(np.exp(gam*L)*(h+k*gam))/(np.exp(-gam*L)*(h-k*gam)+np.exp(gam*L)*(h+k*gam))
    C2 = O/(k*gam)+C1
    Ts = C1*np.exp(-gam*x)+C2*np.exp(gam*x)+Tamb

    return Ts

def sampler(nbrPts):

    point = np.zeros((nbrPts,2))
    point[:,0] = np.random.normal(-18,2,nbrPts)
    point[:,1] = np.random.uniform(0.0018,0.00195,nbrPts)

    return point

def response(point):

    nbrPts = point.shape[0]
    #x = np.linspace(0,70,101)
    x = np.arange(10.0, 70.0, 4.0)
    resp = np.array([fun(x,*point[i]) for i in range(nbrPts)])

    return x,resp

# %% Monte Carlo

nbrPts = int(2e5)
point = sampler(nbrPts)
x,resp = response(point)
mean = np.mean(resp,axis=0)
var = np.var(resp,axis=0)

np.save('mean.npy',mean)
np.save('var.npy',var)

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
plt.plot(x,mean,label='Monte Carlo')
plt.ylabel('Mean [°C]')
plt.xlabel('$x$ [cm]')
plt.legend()

plt.figure(2)
plt.plot(x,var,label='Monte Carlo')
plt.ylabel('Variance [°C$^2$]')
plt.xlabel('$x$ [cm]')
plt.legend()