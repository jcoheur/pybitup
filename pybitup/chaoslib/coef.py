from .tools import timer,printer
import numpy as np

# %% Spectral Projection

def spectral(resp,poly,point,weight=0):
    """Computes the expansion coefficients with spectral projection"""

    printer(0,'Computing coefficients ...')

    V = poly.vander(point)
    if not np.any(weight): weight = 1/V.shape[0]

    # Computes the polynomial chaos coefficients

    V = np.transpose(weight*V.T)
    coef = np.transpose(np.dot(resp.T,V))

    printer(1,'Computing coefficients 100 %')
    return coef

# %% Point Collocation

def colloc(resp,poly,point,weight=0):
    """Computes the expansion coefficients with least-square collocation"""

    printer(0,'Computing coefficients ...')

    resp = np.array(resp)
    shape = (poly[:].shape[0],)+resp.shape[1:]
    resp = resp.reshape(resp.shape[0],-1)
    V = poly.vander(point)

    # Solves the least squares linear system

    if np.any(weight):

        Vt = V.T
        v1 = Vt.dot(np.transpose(weight*Vt))
        v2 = Vt.dot(np.transpose(weight*resp.T))
        coef = np.linalg.solve(v1,v2)

    else: coef = np.linalg.lstsq(V,resp,rcond=None)[0]

    coef = coef.reshape(shape)
    printer(1,'Computing coefficients 100 %')
    return coef

# %% Least Angle Regression

def lars(resp,poly,point,weight=0,it=np.inf):
    """Computes the expansion coefficients with least angle regression"""

    def square(V,resp,weight):
        if np.any(weight):
    
            Vt = V.T
            v1 = Vt.dot(np.transpose(weight*Vt))
            v2 = Vt.dot(np.transpose(weight*resp.T))
            coef = np.linalg.solve(v1,v2)
    
        else: coef = np.linalg.lstsq(V,resp,rcond=None)[0]
        return coef

    # Initialization

    resp = np.array(resp)
    shape = (poly[:].shape[0],)+resp.shape[1:]
    resp = resp.reshape(resp.shape[0],-1)

    V = poly.vander(point)
    nbrResp = resp.shape[1]
    coef = np.zeros((V.shape[1],nbrResp))

    # Standardizes V and calls the lars algorithm

    V1  = V[:,1:]
    stat = [np.mean(V1,axis=0),np.std(V1,axis=0,ddof=1)]
    V1 = (V1-stat[0])/stat[1]

    for i in range(nbrResp):

        coef[:,i] = fit(V1,resp[:,i],stat,it)
        index = np.argwhere(coef[:,i]!=0).flatten()
        coef[index,i] = square(V[:,index],resp[:,i],weight)
        timer(i+1,nbrResp,'Computing coefficients ')

    index = np.argwhere(np.any(coef,axis=1)).flatten()
    coef = coef.reshape(shape)
    return coef,index

# %% Least Angle Regression

def fit(V,resp,stat,it):
    """Internal function of the least angle regression algorithm"""

    # First variable entering the model

    nbrPoly = V.shape[1]
    mean = np.mean(resp)
    coef = np.zeros(nbrPoly+1)

    r = resp-mean
    J = np.atleast_1d(np.argmax(np.abs(np.dot(V.T,r))))
    i = 0

    # Performs the least angle regression iterations

    while (i==0 or alp<1) and (i+1<it):

        alp = 1
        d = np.zeros(nbrPoly)
        Vj = V[:,J]

        u1 = np.dot(Vj.T,Vj)
        u2 = np.dot(Vj.T,r)
        d[J] = np.linalg.solve(u1,u2)

        Vd = np.dot(V,d)
        J = np.append(J,-1)
        u1 = np.dot(V[:,J[0]],r)

        for j in range(nbrPoly):

            alp1 = 1
            if not (j in J):

                u2 = np.dot(V[:,j],Vd)
                u3 = np.dot(V[:,j],r)
                alp1 = (u1-u3)/(u1-u2)

                if not (0<alp1<1):

                    alp1 = 1
                    alp2 = (u1+u3)/(u1+u2)
                    if (0<alp2<1): alp1 = alp2

                if alp1<alp:
                    alp = alp1
                    J[-1] = j

        coef[1:] = coef[1:]+alp*d
        r = r-alp*Vd
        i += 1

    # Translates coefficient back to original scale

    coef[1:] = coef[1:]/stat[1]
    coef[0] = mean-np.dot(coef[1:],stat[0])
    return coef