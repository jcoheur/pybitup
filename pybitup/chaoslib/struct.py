from scipy import sparse
import numpy as np

# %% Polynomial Basis

class Polynomial:
    """Class of orthogonal polynomials basis"""

    def __init__(self,expo,coef,csr=0):

        if sparse.issparse(coef): self.coef = coef.copy()
        else: self.coef = np.copy(np.atleast_2d(np.transpose(coef)).T)
        if csr: self.coef = sparse.csr_matrix(self.coef)

        self.expo = np.copy(np.atleast_2d(expo))
        self.dim = self.expo.shape[0]

    def __getitem__(self,i): return self.coef[i]
    def __setitem__(self,i,coef): self.coef[i] = coef

    # Evaluates the k-th polynomial at the points

    def eval(self,k,point):

        resp = 1
        point = np.reshape(np.transpose(point),(self.dim,-1)).T
        for i in range(self.dim): resp *= np.power(point[:,i,None],self.expo[i])
        resp = self.coef[k].dot(resp.T).flatten()
        return resp

    # Computes the Vandermonde-like matrix of the basis

    def vander(self,point):

        V = 1
        point = np.reshape(np.transpose(point),(self.dim,-1)).T
        for i in range(self.dim): V *= np.power(point[:,i,None],self.expo[i])
        V = np.transpose(self.coef.dot(V.T))
        return V

    # Truncates the polynomial basis at a lower order

    def trunc(self,order):

        nbrPoly = self.coef.shape[0]
        id1 = np.where(np.sum(self.expo,axis=0)<=order)[0]
        id2 = np.setdiff1d(np.arange(self.expo.shape[1]),id1)
        id2 = np.setdiff1d(range(nbrPoly),self.coef[:,id2].nonzero()[0])

        self.coef = self.coef[id2]
        self.coef = self.coef[:,id1]
        self.expo = self.expo[:,id1]

        idx = np.unique(self.coef.nonzero()[1])
        self.expo = self.expo[:,idx]
        self.coef = self.coef[:,idx]

    # Removes all the polynomials from the basis except index

    def clean(self,idx):

        self.coef = self.coef[idx]
        idx = np.unique(self.coef.nonzero()[1])
        self.expo = self.expo[:,idx]
        self.coef = self.coef[:,idx]

# %% Expansion Model

class Expansion:
    """Class of polynomial chaos expansion"""

    def __init__(self,coef,poly):

        coef = np.array(coef)
        nbrPoly = poly[:].shape[0]
        shape = (poly[:].shape[1],)+coef.shape[1:]
        coef = coef.reshape(nbrPoly,-1)

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