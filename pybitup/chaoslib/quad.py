from scipy import optimize,linalg
from .tools import printer,timer
from .math import halton
import numpy as np

# %% Quasi-Monte Carlo

def qmcquad(nbrPts,dom,pdf=0):
    """Quadrature rule for uniform quasi-Monte Carlo inregration"""

    dom = np.atleast_2d(dom)
    dim = dom.shape[0]
    point = halton(nbrPts,dim)
    point = np.reshape(point,(-1,dim))

    # Expands the sequence into the provided domain

    for i in range(dim):

        d = dom[i][1]-dom[i][0]
        point[:,i] = d/2+dom[i][0]+d*(point[:,i]-0.5)

    # Computes the weights with the density

    vol = np.prod([dom[i][1]-dom[i][0] for i in range(dim)])
    weight = vol*np.ones(nbrPts)/nbrPts
    point = np.squeeze(point)

    if callable(pdf): weight = np.multiply(weight,pdf(point))
    return point,weight

# %% Tensor Product

def tensquad(nbrPts,dist):
    """Computes the tensor product quadrature rule with recurrence coefficients"""

    if not isinstance(dist,Joint): dist = Joint(dist)

    dim = dist[:].shape[0]
    J = np.zeros((2,nbrPts))
    points = np.zeros((nbrPts,dim))
    weights = np.zeros((nbrPts,dim))

    # Integration points and weights of each variable

    for i in range(dim):

        coef = dist[i].coef(nbrPts)
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
    point = np.squeeze(point)
    return point,weight

# %% Fekete Points

def fekquad(point,poly):
    """Selects the approximate Fekete points and computes their weights"""

    printer(0,"Selecting points ...")
    nbrPoly = poly[:].shape[0]

    # Reconditioning of V and QR factorization

    V = poly.vander(point)
    for i in range(2): V,R = np.linalg.qr(V)
    m = np.sum(V,axis=0)/V.shape[0]

    Q,R,P = linalg.qr(V.T,pivoting=1,mode="economic")
    R = R[:,:nbrPoly]
    q = np.dot(Q.T,m)

    # Computes the weights and Fekete points

    index = P[:nbrPoly]
    weight = linalg.solve_triangular(R,q)

    printer(1,"Selecting points 100 %")
    return index,weight

# %% Positive Quadrature

def nulquad(point,poly):
    """Computes a positive quadrature rule by iterative node removal"""

    V = poly.vander(point)
    nbrPts = V.shape[0]
    index = np.arange(nbrPts)
    nbrIter = nbrPts-V.shape[1]
    weight = np.ones(nbrPts)/nbrPts
    m = np.sum(V,axis=0)/V.shape[0]
    A = V.T

    for i in range(nbrIter):

        timer(i+1,nbrIter,"Selecting points ")
        U,S,Vt = np.linalg.svd(A,full_matrices=True)
        z = Vt[-1]

        # Selects the coefficient to cancel a weight

        wz = weight/z
        idx = np.argmin(np.abs(wz))
        alp = wz[idx]

        # Updates the weights and the matrix

        weight -= alp*z
        weight = np.delete(weight,idx)
        A = np.delete(A,idx,axis=1)
        index = np.delete(index,idx)

    weight = np.linalg.solve(A,m)
    return index,weight

# %% Revised Simplex

def linquad(point,poly):
    """Computes a positive quadrature rule using the revised simplex"""

    printer(0,"Selecting points ...")

    tol = 1e-20
    V = poly.vander(point)
    nbrPts = V.shape[0]
    c = np.ones(nbrPts)
    m = np.sum(V,axis=0)/nbrPts

    # Performs the revised simplex

    x = optimize.linprog(c,A_eq=V.T,b_eq=m,method="revised simplex")
    index = np.argwhere(x['x']>tol).flatten()
    weight = x['x'][index]

    printer(1,"Selecting points 100 %")
    return index,weight