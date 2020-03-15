from scipy import optimize,linalg
from .tools import printer,timer
from .math import rseq
import numpy as np

# %% Quasi-Monte Carlo

def qmcquad(nbrPts,dom,pdf=0):
    """Computes the points and weights for quasi-Monte Carlo inregration"""

    dom = np.atleast_2d(dom)
    dim = dom.shape[0]
    point = rseq(nbrPts,dom)
    weight = np.ones(nbrPts)/nbrPts

    # Computes the weights with the density

    for i in range(dim): weight *= (dom[i][1]-dom[i][0])
    if callable(pdf): weight = np.multiply(weight,pdf(*point.T))
    return point,weight

# %% Tensor Product

def tensquad(nbrPts,lawList):
    """Computes the tensor product quadrature rule with recurrence coefficients"""

    lawList = np.reshape(lawList,-1)
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
    nbrPoly = poly[:].shape[0]

    # Reconditioning of V and LU decomposition

    V = poly.vander(point)
    for i in range(2): V,R = np.linalg.qr(V)
    m = np.sum(V,axis=0)/V.shape[0]

    LU,P,info = linalg.lapack.dgetrf(V)
    L = np.tril(LU[:nbrPoly],-1)
    U = np.triu(LU[:nbrPoly])
    np.fill_diagonal(L,1)

    # Computes the weights and Leja points

    index = P[:nbrPoly]
    u = linalg.solve_triangular(U,m,trans=1)
    weight = linalg.solve_triangular(L.T,u,unit_diagonal=1)

    printer(1,"Selecting points 100 %")
    return index,weight

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

    tol = 1e-25
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

# %% Dual Simplex

def simquad(point,poly):
    """Computes a positive quadrature rule using the dual-simplex"""

    import subprocess as sub
    from scipy import io
    import os

    printer(0,"Selecting points ...")

    V = poly.vander(point)
    m = np.sum(V,axis=0)/V.shape[0]

    path = os.path.dirname(os.path.realpath(__file__))
    io.savemat(path+"\data.mat",mdict={"V":V,"m":m})
    process = sub.Popen([path+"\simplex.exe",path+"\data.mat"],stdout=sub.PIPE)
    weight = process.stdout.readlines()
    os.remove(path+"\data.mat")

    weight = np.array(weight[3:-1],dtype=float)
    index = np.argwhere(weight).flatten()
    weight = weight[index]

    printer(1,"Selecting points 100 %")
    return index,weight