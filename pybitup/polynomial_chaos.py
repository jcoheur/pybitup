from pybitup.distributions import *
import pybitup.chaoslib as cl
from jsmin import jsmin
from json import loads
import numpy as np
import os.path
import sys

# %% Polynomial Chaos Expansion Wrapper

class PCE:
    """Wrapper for chaoslib, the class takes a json file containing the polynomial expansion parameters"""

    def __init__(self,parameters): self.param = parameters
    def save_pickle(self,item,name): cl.save(item,name)

    # %% Computes the polynomials

    def compute_polynomials(self,point,weight):

        method = self.param["polynomials"]["method"]
        order = self.param["polynomials"]["order"]
        trunc = self.param["polynomials"]["hyperbolic_truncation"]

        if method=="gram_schmidt": return cl.gschmidt(order,point,weight,trunc)
        elif method=="recurrence":

            dist = []
            for law in self.param["polynomials"]["parameter_laws"]:

                name = list(law.keys())[0]
                param = law[name]

                if name=="uniform": dist.append(Uniform(param,0))
                elif name=="normal": dist.append(Gaussian(param,0))
                elif name=="gamma": dist.append(Gamma(param,0))
                elif name=="beta": dist.append(Beta(param,0))
                elif name=="expo": dist.append(Exponential(param,0))
                elif name=="lognorm": dist.append(Lognormal(param,0))
                else: raise Exception("compute_polynomials: unknown law")
                
            return cl.polyrecur(order,dist,trunc)

        else: raise Exception("compute_polynomials: unknown method")

    # %% Selects the coefficients

    def compute_coefficients(self,resp,poly,point,weight):

        method = self.param["coefficients"]["method"]

        if method=="lars":

            it = self.param["coefficients"]["iterations"]
            if it=="unlimited": it = np.inf
            return cl.lars(resp,poly,point,weight,it)
        
        elif method=="lasso":

            it = self.param["coefficients"]["iterations"]
            if it=="unlimited": it = np.inf
            return cl.lasso(resp,poly,point,weight,it)

        elif method=="spectral": return cl.spectral(resp,poly,point,weight)
        elif method=="colloc": return cl.colloc(resp,poly,point,weight)
        else: raise Exception("compute_coefficients: unknown method")

    # %% Computes the quadrature

    def compute_quadrature(self,point,poly):

        # Monte Carlo

        if self.param["quadrature"]["method"]=="monte_carlo": weight = None

        # Recurrence coefficients

        elif self.param["quadrature"]["method"]=="recurrence":

            dist = []
            order = self.param["quadrature"]["order_quadrature"]
            for law in self.param["polynomials"]["parameter_laws"]:

                name = list(law.keys())[0]
                param = law[name]
    
                if name=="uniform": dist.append(Uniform(param,0))
                elif name=="normal": dist.append(Gaussian(param,0))
                elif name=="gamma": dist.append(Gamma(param,0))
                elif name=="beta": dist.append(Beta(param,0))
                elif name=="expo": dist.append(Exponential(param,0))
                elif name=="lognorm": dist.append(Lognormal(param,0))
                else: raise Exception("compute_quadrature: unknown law")

            point,weight = cl.tensquad(order,dist)
            
        # Weakly admissible mesh

        else:
            method = self.param["quadrature"]["method"]
        
            if method=="fekete": index,weight = cl.fekquad(point,poly)
            elif method=="simplex": index,weight = cl.simquad(point,poly)
            elif method=="iterative": index,weight = cl.nulquad(point,poly)
            else: raise Exception("compute_quadrature: unknown method")

            poly.trunc(self.param["quadrature"]["order_truncation"])
            point = point[index]

        return point,weight

    # %% Evaluates the function

    def function_evaluator(self,function,point):

        point = np.atleast_2d(np.transpose(point)).T
        nbrPts = point.shape[0]
        resp = []

        for i in range(nbrPts):

            function.run_model(point[i])
            resp.append(function.model_eval)

        return np.array(resp)

    # %% Computes the polynomial chaos expansion

    def compute_pce(self,function):

        # Stores the points and eventual weights

        pointFile = self.param["polynomials"]["point_coordinates"]

        if os.path.splitext(pointFile)[1]==".npy": point = np.load(pointFile)
        elif os.path.splitext(pointFile)[1]==".csv": point = np.loadtxt(pointFile,delimiter=",")
        else: raise Exception("point_coordinates file not found")

        weightFile = self.param["polynomials"]["point_weights"]

        if os.path.splitext(weightFile)[1]==".npy": weight = np.load(weightFile)
        elif os.path.splitext(weightFile)[1]==".csv": weight = np.loadtxt(weightFile,delimiter=",")
        else: weight = None

        # Computes the pce elements

        poly = self.compute_polynomials(point,weight)
        point,weight = self.compute_quadrature(point,poly)
        resp = self.function_evaluator(function,point)
        coef = self.compute_coefficients(resp,poly,point,weight)

        if isinstance(coef,tuple):

            index = coef[1]
            coef = coef[0]
            poly.clean(index)
            coef = coef[index]

        # Computes the pce model

        model = cl.Expansion(coef,poly)
        return poly,coef,model