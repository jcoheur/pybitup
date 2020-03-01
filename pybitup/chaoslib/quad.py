from .math import halton,rseq
from .tools import printer
from scipy import linalg
import numpy as np

# %% Quasi-Monte Carlo

def qmcquad(nbrPts,dom,pdf=0):
    """Computes the points and weights for quasi-Monte Carlo inregration"""

    dom = np.atleast_2d(dom)
    dim = dom.shape[0]

    point = rseq(nbrPts,dom)
    weight = np.ones(nbrPts)/nbrPts

    # Computes the weights with probability density function

    for i in range(dim): weight *= (dom[i][1]-dom[i][0])
    if callable(pdf): weight = np.multiply(weight,pdf(*point.T))
    return point,weight

# %% Tensor Product

def tensquad(nbrPts,lawList):
    """Computes the tensor product quadrature rule with recurrence coefficients"""

    lawList = np.atleast_1d(lawList)
    dim = lawList.shape[0]
    J = np.zeros((2,nbrPts))
    points = np.zeros((nbrPts,dim))
    weights = np.zeros((nbrPts,dim))

    # Integration points and weights of each variable

    for i in range(dim):

        coef = lawList[i].coef(nbrPts)
        J[1] = np.append(np.sqrt(coef[1][1:]),[0])
        J[0] = coef[0]

        val,vec = linalg.eig_banded(J,lower=1)
        weights[:,i] = vec[0,:]**2
        points[:,i] = val.real

    # Places the points and weights in the domain

    point = np.zeros((nbrPts**dim,dim))
    weight = np.zeros((nbrPts**dim,dim))

    for i in range(dim):

        v1 = np.repeat(points[:,i],nbrPts**(dim-1-i))
        v2 = np.repeat(weights[:,i],nbrPts**(dim-1-i))
        weight[:,i] = np.tile(v2,nbrPts**i)
        point[:,i] = np.tile(v1,nbrPts**i)

    weight = np.prod(weight,axis=1)
    return point,weight

# %% Leja Points

def lejquad(point,poly):
    """Selects the discrete Leja points and computes their weights"""

    printer(0,"Selecting points ...")

    nbrPoly = poly.nbrPoly
    m = np.zeros(nbrPoly)
    m[0] = 1

    # Reconditioning of V and LU decomposition

    V = poly.vander(point)
    for i in range(2): V,R = np.linalg.qr(V)

    LU,P,info = linalg.lapack.dgetrf(V)
    L = np.tril(LU[:nbrPoly],-1)
    U = np.triu(LU[:nbrPoly])
    np.fill_diagonal(L,1)

    # Computes the weights and Leja points

    index = P[:nbrPoly]
    u = linalg.solve_triangular(U,m,trans=1)
    weight = linalg.solve_triangular(L.T,u,unit_diagonal=1)
    weight = weight/np.sum(weight)

    printer(1,"Selecting points 100 %")
    return index,weight

# %% Fekete Points

def fekquad(point,poly):
    """Selects the approximate Fekete points and computes their weights"""

    printer(0,"Selecting points ...")

    nbrPoly = poly.nbrPoly
    m = np.zeros(nbrPoly)
    m[0] = 1

    # Reconditioning of V and QR factorization

    V = poly.vander(point)
    for i in range(2): V,R = np.linalg.qr(V)

    Q,R,P = linalg.qr(V.T,pivoting=1,mode='economic')
    R = R[:,:nbrPoly]
    q = np.dot(Q.T,m)

    # Computes the weights and Fekete points

    index = P[:nbrPoly]
    weight = linalg.solve_triangular(R,q)
    weight = weight/np.sum(weight)

    printer(1,"Selecting points 100 %")
    return index,weight