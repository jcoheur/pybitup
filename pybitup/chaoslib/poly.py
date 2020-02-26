from .math import indextens,tensdot
from .struct import Polynomial
from .tools import printer
from scipy import sparse
import numpy as np

# %% Gram-Schmidt

def gschmidt(order,point,weight=0,trunc=1):
    """Computes the orthonormal polynomial basis using Gram-Schmidt"""

    printer(0,"Computing polynomials ...")

    point = np.atleast_2d(np.transpose(point)).T
    if not np.any(weight): weight = 1/point.shape[0]
    dim = point.shape[1]

    # Creates the tensor product of univariate polynomials

    nbrPoly = order+1
    expo = indextens(order,dim,trunc)
    nbrPoly = expo.shape[1]
    coef = sparse.eye(nbrPoly)
    norm = np.ones(nbrPoly)
    base = Polynomial(expo,coef,norm)

    # Computes modified Gram-Schmidt algorithm

    V = base.vander(point)
    V = np.transpose(np.sqrt(weight)*V.T)
    R = np.linalg.qr(V,"r")
    coef = np.linalg.inv(R).T
    coef[0,0] = 1

    poly = Polynomial(expo,coef,norm,1)
    printer(1,"Computing polynomials 100 %")
    return poly

# %% Recurrence Coefficients

def polyrecur(order,lawList,trunc=1):
    """Computes the orthogonal polynomial basis using recurrence coefficients"""

    printer(0,"Computing polynomials ...")

    nbrPoly = order+1
    lawList = np.atleast_1d(lawList)
    dim = lawList.shape[0]
    norm = np.ones(nbrPoly)
    expo = np.arange(nbrPoly)
    coef = np.zeros((nbrPoly,nbrPoly))
    coef[0,0] = 1
    polyList = []

    # Creates the univariate polynomial basis

    for i in range(dim):

        polyList.append(Polynomial(expo,coef,norm))
        AB = lawList[i].coef(nbrPoly)

        for j in range(1,nbrPoly):

            polyList[i].norm[j] = polyList[i].norm[j-1]*AB[1,j]
            polyList[i][j] = np.roll(polyList[i][j-1],1,axis=0)
            polyList[i][j] -= AB[0,j-1]*polyList[i][j-1]+AB[1,j-1]*polyList[i][j-2]

    # Performs the basis tensor product

    poly = tensdot(polyList,order,trunc)
    printer(1,"Computing polynomials 100 %")
    return poly