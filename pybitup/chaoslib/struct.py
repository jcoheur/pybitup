from scipy import sparse
import numpy as np

# %% Polynomial Basis

class Polynomial:
    """Class of orthogonal polynomials basis"""

    def __init__(self,expo,coef,norm,csr=0):

        if sparse.issparse(coef): self.coef = coef.copy()
        else: self.coef = np.atleast_2d(np.transpose(coef)).T.copy()
        if csr: self.coef = sparse.csr_matrix(self.coef)

        self.expo = np.atleast_2d(expo).copy()
        self.norm = np.atleast_1d(norm).copy()
        self.nbrPoly = self.coef.shape[0]
        self.dim = self.expo.shape[0]

    def __getitem__(self,i): return self.coef[i]
    def __setitem__(self,i,coef): self.coef[i] = coef

    # Evaluates the i-th polynomial at the points

    def eval(self,i,point):

        if self.dim>1: point = np.atleast_2d(point)
        else: point = np.atleast_2d(np.transpose(point)).T

        resp = np.power(point[:,0,None],self.expo[0])
        for i in range(1,self.dim): resp *= np.power(point[:,i,None],self.expo[i])
        resp = self.coef[i].dot(resp.T).flatten()
        return resp

    # Computes the Vandermonde-like matrix of the basis

    def vander(self,point):

        if self.dim>1: point = np.atleast_2d(point)
        else: point = np.atleast_2d(np.transpose(point)).T

        V = np.power(point[:,0,None],self.expo[0])
        for i in range(1,self.dim): V *= np.power(point[:,i,None],self.expo[i])
        V = np.transpose(self.coef.dot(V.T))
        return V

    # Truncates the polynomial basis at a lower order

    def trunc(self,order):

        id1 = np.where(np.sum(self.expo,axis=0)<=order)[0]
        id2 = np.setdiff1d(np.arange(self.expo.shape[1]),id1)
        id2 = np.setdiff1d(range(self.nbrPoly),self.coef[:,id2].nonzero()[0])

        self.coef = self.coef[id2]
        self.norm = self.norm[id2]
        self.coef = self.coef[:,id1]
        self.expo = self.expo[:,id1]
        self.nbrPoly = self.coef.shape[0]

        idx = np.unique(self.coef.nonzero()[1])
        self.expo = self.expo[:,idx]
        self.coef = self.coef[:,idx]

    # Removes all the polynomials from the basis except index

    def clean(self,idx):

        self.coef = self.coef[idx]
        self.norm = self.norm[idx]
        self.nbrPoly = self.coef.shape[0]

        idx = np.unique(self.coef.nonzero()[1])
        self.expo = self.expo[:,idx]
        self.coef = self.coef[:,idx]

# %% Expansion Model

class Expansion:
    """Class of polynomial substitution model"""

    def __init__(self,coef,poly):

        coef = np.atleast_2d(np.transpose(coef)).T
        model = coef[0,None].T*poly[0]
        for i in range(1,poly.nbrPoly): model += coef[i,None].T*poly[i]

        self.expo = np.atleast_2d(poly.expo).copy()
        self.dim = self.expo.shape[0]
        self.coef = model

    # Evaluates the expansion at the points

    def eval(self,point):

        if self.dim>1: point = np.atleast_2d(point)
        else: point = np.atleast_2d(np.transpose(point)).T

        V = np.power(point[:,0,None],self.expo[0])
        for i in range(1,self.dim): V *= np.power(point[:,i,None],self.expo[i])
        V = np.dot(V,self.coef.T)
        return V