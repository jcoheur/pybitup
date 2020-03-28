from sobol_seq import i4_sobol_generate
from scipy import linalg
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

    index = np.transpose(index)
    return index

# %% Prime Numbers

def prime(nbrPrime):
    """Generates an array containing the first prime numbers"""

    def check(nbr,i=5):
        while i<=np.sqrt(nbr):
            if (nbr%i==0) or nbr%(i+2)==0: return 0
            else: i += 6

    # Appends the prime numbers to the list

    prime = [2]
    nbr = 3

    while len(prime)<nbrPrime:

        if (nbr==3): prime.append(nbr)
        elif (nbr%2==0) or (nbr%3==0): pass
        elif check(nbr)!=0: prime.append(nbr)
        nbr += 2

    prime = np.array(prime)
    return prime

# %% Sobol Sequence

def sobol(nbrPts,dim=1):
    """Computes the Sobol sequence of quasi-random numbers"""

    point = i4_sobol_generate(dim,nbrPts)
    point = np.squeeze(point)
    return point

# %% Halton Sequence

def halton(nbrPts,dim=1):
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

    # Computes Halton sequence using Van der Corput

    pr = prime(dim)
    point = np.zeros((nbrPts,dim))
    indices = [idx+pr[-1] for idx in range(nbrPts)]
    for i in range(dim): point[:,i] = vdcseq(indices,pr[i])

    point = np.squeeze(point)
    return point

# %% R-Sequence

def rseq(nbrPts,dim=1):
    """Computes the R-sequence of quasi-random numbers"""

    phi = 2
    alpha = np.zeros(dim)
    point = np.zeros((nbrPts,dim))

    # Generates the points in the unit hypercube

    for i in range(10): phi = (1+phi)**(1/(dim+1))
    for i in range(dim): alpha[i] = (1/phi)**(1+i)%1
    for i in range(nbrPts): point[i] = (0.5+alpha*(1+i))%1

    point = np.squeeze(point)
    return point

# %% Quadratic Resampling

def match(point,mean,cov):
    """Performs a quadratic resampling according to imposed moments"""

    sCov = np.cov(point.T)
    sMean = np.mean(point,axis=0)
    v2 = np.linalg.inv(linalg.sqrtm(sCov))
    v1 = linalg.sqrtm(cov)
    H = np.dot(v1,v2)

    point = (point-sMean).dot(H.T)+mean
    return point

# %% Monte Carlo Sampler

def random(nbrPts,dom,pdf):
    """Generates a sample of points according to a probability distribution"""

    dom = np.atleast_2d(dom)
    dim = dom.shape[0]

    y = np.random.uniform(0,1,nbrPts)
    x = np.array([np.random.uniform(a,b,nbrPts) for [a,b] in dom])
    index = np.argwhere(y<pdf(*x)).flatten()
    point = np.transpose(x[:,index])

    # Repeats the operation to obtain the desired size

    while (point.shape[0]<nbrPts):

        y = np.random.uniform(0,1,nbrPts)
        x = np.array([np.random.uniform(a,b,nbrPts) for [a,b] in dom])
        index = np.argwhere(y<pdf(*x)).flatten()

        x = np.transpose(x[:,index])
        point = np.concatenate((point,x),axis=0)

    # Removes the overflow of points

    index = np.random.choice(point.shape[0],nbrPts)
    point = point[index]
    return point

# %% Statistical Moments

def stats(coef,poly,sobol=0):
    """Computes the first statistical moments and the Sobol coefficients"""

    mean = coef[0]
    var = np.sum(coef[1:]**2,axis=0)

    if sobol:

        S,St = [],[]
        expo = poly.expo
        dim = expo.shape[0]
        nbrPoly = poly[:].shape[0]

        # Computes the Sobol coefficients

        for i in range(dim):
    
            order = np.sum(expo,axis=0)
            pIdx = np.array([poly[i].nonzero()[-1][-1] for i in range(nbrPoly)])
    
            sIdx = np.where(expo[i]-order==0)[0].flatten()[1:]
            index = np.where(np.in1d(pIdx,sIdx))[0]
            S.append(np.sum(coef[index]**2)/var)
    
            sIdx = np.where(expo[i])[0].flatten()
            index = np.where(np.in1d(pIdx,sIdx))[0]
            St.append(np.sum(coef[index]**2)/var)
    
        S = np.transpose(S)
        St = np.transpose(St)

        return mean,var,S,St
    else: return mean,var

# %% PCA Whitening

class Pca:
    """Class of PCA whitening for linearly separable variables"""

    def __init__(self,point):

        self.mean = np.mean(point,axis=0)
        self.std = np.std(point,axis=0,ddof=1)

        # Standardizes and computes the whitening matrix

        point = (point-self.mean)/self.std
        cov = np.cov(point.T)
        val,vec = np.linalg.eig(cov)

        self.A = np.diag(np.sqrt(1/val)).dot(vec.T)
        self.invA = np.linalg.inv(self.A)

    # Mapping from correlated to whitened

    def white(self,point):

        point = np.transpose((point-self.mean)/self.std)
        point = np.transpose(self.A.dot(point))
        return point

    # Mapping from whitened to correlated

    def corr(self,point):

        point = np.dot(self.invA,point.T)
        point = self.std*point.T+self.mean
        return point