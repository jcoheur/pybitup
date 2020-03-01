from .tools import timer,printer
from .struct import Expansion
from .poly import polyrecur
from scipy import integrate
import numpy as np

# %% Spectral Projection

def spectral(resp,poly,point,weight=0):
    """Computes the expansion coefficients with spectral projection"""

    printer(0,"Computing coefficients ...")

    V = poly.vander(point)
    if not np.any(weight): weight = 1/V.shape[0]
    resp = np.atleast_2d(np.transpose(resp)).T
    nbrResp = resp.shape[1]

    # Computes the polynomial chaos coefficients

    V = np.transpose(weight*V.T)
    coef = np.transpose(np.dot(resp.T,V))
    printer(1,"Computing coefficients 100 %")
    return coef

# %% Point Collocation

def colloc(resp,poly,point,pdf=0):
    """Computes the expansion coefficients with least-square collocation"""

    printer(0,"Computing coefficients ...")
    resp = np.atleast_2d(np.transpose(resp)).T
    V = poly.vander(point)

    # If weighted point collocation

    if callable(pdf):

        point = np.atleast_2d(np.transpose(point))
        weight = np.sqrt(pdf(*point))
        V = np.transpose(weight*V.T)
        resp = np.transpose(weight*resp.T)

    # Solves the least squares linear system

    coef = np.linalg.lstsq(V,resp,rcond=None)[0]
    printer(1,"Computing coefficients 100 %")
    return coef

# %% Least Angle Regression

def lars(resp,poly,point,it=np.inf):
    """Performs the least angle regression algorithm"""

    def fit(V,resp):

        # First variable entering the model

        respMean = np.mean(resp)
        r = resp-respMean
        coef = np.zeros(nbrPoly)
        J = np.atleast_1d(np.argmax(abs(np.dot(V.T,r))))
        i = 0

        # Least angle regression algorithm

        while (i==0 or alp<1) and (i+1<it):

            alp = 1
            d = np.zeros(nbrPoly-1)
            Vj = V[:,J]

            u1 = np.dot(Vj.T,Vj)
            u2 = np.dot(Vj.T,r)
            d[J] = np.linalg.solve(u1,u2)

            Vd = np.dot(V,d)
            J = np.append(J,-1)
            u1 = np.dot(V[:,J[0]],r)

            for j in range(nbrPoly-1):
                alp1 = 1
    
                if not (j in J):
                    u2 = np.dot(V[:,j],Vd)
                    den = u1-u2
    
                    if abs(den)>tol:
                        u3 = np.dot(V[:,j],r)
                        alp1 = (u1-u3)/den
    
                        if not (tol<alp1<1-tol):
                            den = u1+u2
                            alp1 = 1

                            if abs(den)>tol:

                                alp2 = (u1+u3)/den
                                if (tol<alp2<1-tol): alp1 = alp2

                        if alp1+tol<alp:
                            alp = alp1
                            J[-1] = j

            coef[1:] = coef[1:]+alp*d
            r = r-alp*Vd
            i += 1

        # Translates coefficient back to original scale
    
        coef[1:] = coef[1:]/Vstd
        coef[0] = respMean-np.dot(coef[1:],Vmean)
        return coef

    resp = np.atleast_2d(np.transpose(resp)).T
    nbrResp = resp.shape[1]
    nbrPoly = poly.nbrPoly
    coef = np.zeros((nbrPoly,nbrResp))
    V = poly.vander(point)
    tol = 1e-8

    # Standardizes the Vandermonde matrix

    V1  = V[:,1:]
    Vstd = np.std(V1,axis=0,ddof=1)
    Vmean = np.mean(V1,axis=0)
    V1 = (V1-Vmean)/Vstd

    for i in range(nbrResp):

        coef[:,i] = fit(V1,resp[:,i])
        index = np.argwhere(coef[:,i]!=0).flatten()
        coef[index,i] = np.linalg.lstsq(V[:,index],resp[:,i],rcond=None)[0]
        timer(i+1,nbrResp,"Computing coefficients ")

    index = np.argwhere(np.any(coef,axis=1)).flatten()
    return coef,index

# %% Polynomial Chaos

def transfo(invcdf,order,law):
    """Maps an arbitrary random variable to another distribution"""

    nbrPoly = order+1
    coef = np.zeros(nbrPoly)
    poly = polyrecur(order,law)

    # Computes polynomial chaos coefficients and model

    for i in range(nbrPoly):

        fun = lambda x: invcdf(x)*poly.eval(i,law.invcdf(x))
        coef[i] = integrate.quad(fun,0,1)[0]

    expan = Expansion(coef,poly)
    transfo = lambda x: expan.eval(x)
    return transfo