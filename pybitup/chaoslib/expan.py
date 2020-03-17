from .poly import polyrecur
from scipy import integrate
import numpy as np

# %% PCE Model

class Expansion:
    """Class of polynomial chaos expansion"""

    def __init__(self,coef,poly):

        coef = np.array(coef)
        shape = (poly[:].shape[1],)+coef.shape[1:]
        coef = coef.reshape(poly[:].shape[0],-1)

        self.expo = np.copy(np.atleast_2d(poly.expo))
        self.coef = poly[:].T.dot(coef).reshape(shape).T
        self.dim = self.expo.shape[0]

    # Evaluates the expansion at the points

    def eval(self,point):

        V = 1
        point = np.reshape(np.transpose(point),(self.dim,-1)).T
        for i in range(self.dim): V *= np.power(point[:,i,None],self.expo[i])
        V = np.squeeze(np.dot(self.coef,V.T).T)
        return V

# %% Univariate Expansion

def transfo(invcdf,order,dist):
    """Maps an arbitrary random variable to another distribution"""

    nbrPoly = order+1
    coef = np.zeros(nbrPoly)
    poly = polyrecur(order,dist)

    # Computes polynomial chaos coefficients and model

    for i in range(nbrPoly):

        fun = lambda x: invcdf(x)*poly.eval(i,dist.invcdf(x))
        coef[i] = integrate.quad(fun,0,1)[0]

    expan = Expansion(coef,poly)
    transfo = lambda x: expan.eval(x)
    return transfo