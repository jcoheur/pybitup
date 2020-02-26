from .struct import Polynomial
import numpy as np

# %% Multi-index

def indextens(order,dim,trunc):
    """Computes the multi-index matrix of polynomial tensor product"""

    nbrPoly = order+1
    base = np.arange(nbrPoly)[:,None]
    index = base.copy()

    # Creates the matrix of index

    for i in range(1,dim):

        v1 = np.tile(index,(nbrPoly,1))
        v2 = np.repeat(base,v1.shape[0]/nbrPoly,axis=0)

        index = np.concatenate((v2,v1),axis=1)
        norm = np.sum(index**trunc,axis=1)**(1/trunc)
        index = index[norm.round(6)<=order]
        index = index[np.argsort(np.sum(index,axis=-1))]

    return index.T

# %% Tensor Product

def tensdot(polyList,order,trunc):
    """Computes the tensor product of univariate polynomial basis"""

    def reshape(poly,expo):

        poly.coef = poly[:][:,expo]
        poly.expo = expo
        return poly

    dim = len(polyList)
    expo = indextens(order,dim,trunc)
    nbrPoly = expo.shape[1]
    coef = np.eye(nbrPoly)
    norm = np.ones(nbrPoly)

    for i in range(dim):
        polyList[i] = reshape(polyList[i],expo[i])

    # Tensor product of the univariate basis

    for i in range(nbrPoly):

        norm[i] = np.prod([polyList[j].norm[expo[j,i]] for j in range(dim)])
        coef[i] = np.prod([polyList[j][expo[j,i]] for j in range(dim)],axis=0)

    poly = Polynomial(expo,coef,norm,1)
    return poly

# %% Prime Numbers

def prime(nbrPrime):
    """Computes a list of first prime numbers"""

    def check(nbr,i=5):
        while i<=np.sqrt(nbr):
            if (nbr%i==0) or nbr%(i+2)==0: return 0
            i += 6

    # Appends the prime numbers to the list

    prime = [2]
    nbr = 3

    while len(prime)<nbrPrime:

        if (nbr==3): prime.append(nbr)
        elif (nbr%2==0) or (nbr%3==0): pass
        elif check(nbr)!=0: prime.append(nbr)
        nbr += 2

    return prime

# %% Halton Sequence

def halton(nbrPts,dom):
    """Computes the Halton sequence of quasi-random numbers"""

    def vdcseq(idx,nbrBase):

        idx = 1+np.array(idx).flatten()
        active = np.ones(idx.shape[0],dtype=bool)
        point = np.zeros(idx.shape[0])
        base = nbrBase

        while np.any(active):
    
            point[active] += (idx[active]%nbrBase)/base
            idx //= nbrBase
            base *= nbrBase
            active = idx>0
    
        return point

    # Halton sequence using Van der Corput

    dom = np.atleast_2d(dom)
    dim = dom.shape[0]
    pr = prime(dim)
    point = np.zeros((nbrPts,dim))
    indices = [idx+pr[-1] for idx in range(nbrPts)]
    for i in range(dim): point[:,i] = vdcseq(indices,pr[i])

    # Expands the hypercube into the provided domain

    for i in range(dim):

        width = dom[i][1]-dom[i][0]
        point[:,i] = (point[:,i]-0.5)*width+width/2+dom[i][0]

    return point

# %% R-sequence

def rseq(nbrPts,dom):
    """Computes the R-sequence of quasi-random numbers"""

    phi = 2
    dom = np.atleast_2d(dom)
    dim = dom.shape[0]
    alpha = np.zeros(dim)
    point = np.zeros((nbrPts,dim))

    # Generates the points in the unit hypercube

    for i in range(10): phi = (1+phi)**(1/(dim+1))
    for i in range(dim): alpha[i] = (1/phi)**(1+i)%1
    for i in range(nbrPts): point[i] = (0.5+alpha*(1+i))%1

    # Expands the hypercube into the provided domain

    for i in range(dim):

        width = dom[i][1]-dom[i][0]
        point[:,i] = (point[:,i]-0.5)*width+width/2+dom[i][0]

    return point