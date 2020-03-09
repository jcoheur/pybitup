from scipy import special
import numpy as np

# %% Uniform Law

class Uniform:
    """Class of uniform law with lower and upper boundaries"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.invcdf = lambda x: (b-a)*x+a
        self.pdf = lambda x: x**0/(b-a)
        self.sampler = lambda x: np.random.uniform(a,b,x)

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0].fill((b+a)/2)
        coef[1] = ((b-a)*n/2)**2/(4*n**2-1)
        return coef

# %% Normal Law

class Normal:
    """Class of normal law with mean and standard deviation"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.invcdf = lambda x: a+np.sqrt(2)*b*special.erfinv(2*x-1)
        self.pdf = lambda x: np.exp(-0.5*((x-a)/b)**2)/(b*np.sqrt(2*np.pi))
        self.sampler = lambda x: np.random.normal(a,b,x)

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0].fill(a)
        coef[1] = b**2*n
        return coef

# %% Exponential Law

class Expo:
    """Class of exponential law with inverse scale"""

    def __init__(self,a):

        self.arg = a
        self.invcdf = lambda x: -np.log(1-x)/a
        self.pdf = lambda x: a*np.exp(-a*x)
        self.sampler = lambda x: np.random.exponential(a,x)

    def coef(self,nbrCoef):

        a = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = a*(1+2*n)
        coef[1] = (a*n)**2
        return coef

# %% Gamma Law

class Gamma:
    """Class of gamma law with shape and scale"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.invcdf = lambda x: b*special.gammaincinv(a,x)
        self.pdf = lambda x: x**(a-1)*np.exp(-x/b)/(special.gamma(a)*b**a)
        self.sampler = lambda x: np.random.gamma(a,b,x)

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = (2*n+a)*b
        coef[1] = (n+a-1)*n*b**2
        return coef

# %% Lognormal Law

class Lognorm:
    """Class of lognormal law with mean and variance"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.invcdf = lambda x: np.exp(a+np.sqrt(2)*b*special.erfinv(2*x-1))
        self.pdf = lambda x: np.exp(-0.5*((np.log(x)-a)/b)**2)/(x*b*np.sqrt(2*np.pi))
        self.sampler = lambda x: np.random.lognormal(a,b,x)

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = (np.exp((n+1)*b**2)+np.exp(n*b**2)-1)*np.exp(((2*n-1)*b**2)/2+a)
        coef[1] = (np.exp(n*b**2)-1)*np.exp((3*n-2)*b**2+2*a)
        return coef

# %% Beta Law

class Beta:
    """Class of beta law with shape parameters"""

    def __init__(self,a,b):

        self.arg = [a,b]
        self.invcdf = lambda x: special.betaincinv(a,b,x)
        self.pdf = lambda x: x**(a-1)*(1-x)**(b-1)/special.beta(a,b)
        self.sampler = lambda x: np.random.beta(a,b,x)

    def coef(self,nbrCoef):

        [a,b] = self.arg
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))

        nab = 2*n+a+b
        B1 = a*b*1./((a+b+1)*(a+b)**2)
        B2 = (n+a-1)*(n+b-1)*n*(n+a+b-2)/((nab-1)*(nab-3)*(nab-2)**2+2*((n==0)+(n==1)))
        coef[0] = ((a-1)**2-(b-1)**2)*0.5/(nab*(nab-2)+(nab==0)+(nab==2))+0.5
        coef[1] = np.where((n==0)+(n==1),B1,B2)
        return coef

# %% PCA Whitening

class Pca:
    """Class of PCA whitening for linearly correlated variables"""

    def __init__(self,point):

        self.mean = np.mean(point,axis=0)
        self.std = np.std(point,axis=0,ddof=1)

        # Standardizes and computes the whitening matrix

        point = (point-self.mean)/self.std
        cov = np.cov(point.T)
        val,vec = np.linalg.eig(cov)

        self.A = np.diag(np.sqrt(1/val)).dot(vec.T)
        self.invA = np.linalg.inv(self.A)

    def decor(self,point):

        point = np.transpose((point-self.mean)/self.std)
        point = np.transpose(self.A.dot(point))
        return point

    def cor(self,point):

        point = np.dot(self.invA,point.T)
        point = self.std*point.T+self.mean
        return point