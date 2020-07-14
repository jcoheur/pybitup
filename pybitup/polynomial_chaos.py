from pybitup.distributions import *
import pybitup.pce as pce
from jsmin import jsmin
from json import loads
import numpy as np
import os.path
import sys

# %% Polynomial Chaos Expansion Wrapper

class PCE:
    """Wrapper for chaoslib, the class takes a json file containing the polynomial expansion parameters"""

    def __init__(self,input_pce,parameters): 
        self.input_pce = input_pce
        self.param = parameters
        self.dist = []

        for param_name in self.param.keys():

            if (self.param[param_name].get("distribution") is not None):
                name = self.param[param_name]["distribution"]
                param = self.param[param_name]["hyperparameters"]
                
                if name=="uniform": self.dist.append(Uniform(param,0))
                elif name=="normal": self.dist.append(Gaussian(param,0))
                elif name=="gamma": self.dist.append(Gamma(param,0))
                elif name=="beta": self.dist.append(Beta(param,0))
                elif name=="expo": self.dist.append(Exponential(param,0))
                elif name=="lognorm": self.dist.append(Lognormal(param,0))
                else: raise Exception("compute_quadrature or compute_polynomials: unknown law")

    def save_pickle(self,item,name): pce.save(item,name)

    # %% Computes the polynomials

    def compute_polynomials(self,point,weight):

        method = self.input_pce["polynomials"]["method"]
        order = self.input_pce["polynomials"]["order"]
        trunc = self.input_pce["polynomials"]["hyperbolic_truncation"]

        if method=="gram_schmidt": return pce.gschmidt(order,point,weight,trunc)
        elif method=="recurrence": return pce.polyrecur(order,self.dist,trunc)
        else: raise Exception("compute_polynomials: unknown method")

    # %% Selects the coefficients

    def compute_coefficients(self,resp,poly,point,weight):

        method = self.input_pce["coefficients"]["method"]

        if method=="lars":

            it = self.input_pce["coefficients"]["iterations"]
            if it=="unlimited": it = np.inf
            return pce.lars(resp,poly,point,weight,it)
        
        elif method=="lasso":

            it = self.input_pce["coefficients"]["iterations"]
            if it=="unlimited": it = np.inf
            return pce.lasso(resp,poly,point,weight,it)

        elif method=="spectral": return pce.spectral(resp,poly,point,weight)
        elif method=="colloc": return pce.colloc(resp,poly,point,weight)
        else: raise Exception("compute_coefficients: unknown method")

    # %% Computes the quadrature

    def compute_quadrature(self,point,poly):

        # Monte Carlo

        if self.input_pce["quadrature"]["method"]=="monte_carlo": weight = None

        # Recurrence coefficients

        elif self.input_pce["quadrature"]["method"]=="recurrence":

            order = self.input_pce["quadrature"]["order_quadrature"]
            point,weight = pce.tensquad(order,self.dist)
            
        # Weakly admissible mesh

        else:
            method = self.input_pce["quadrature"]["method"]
        
            if method=="fekete": index,weight = pce.fekquad(point,poly)
            elif method=="simplex": index,weight = pce.simquad(point,poly)
            elif method=="iterative": index,weight = pce.nulquad(point,poly)
            else: raise Exception("compute_quadrature: unknown method")

            poly.trunc(self.input_pce["quadrature"]["order_truncation"])
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

        try:
            pointFile = self.param["polynomials"]["point_coordinates"]
            if pointFile=="None": point = None
            elif os.path.splitext(pointFile)[1]==".npy": point = np.load(pointFile)
            elif os.path.splitext(pointFile)[1]==".csv": point = np.loadtxt(pointFile,delimiter=",")
            else: raise Exception("point_coordinates file not found")
            
        except: point = None

        try:
            weightFile = self.param["polynomials"]["point_weights"]
            if weightFile=="None": weight = None
            if os.path.splitext(weightFile)[1]==".npy": weight = np.load(weightFile)
            elif os.path.splitext(weightFile)[1]==".csv": weight = np.loadtxt(weightFile,delimiter=",")
            else: raise Exception("point_weights file not found")
            
        except: weight = None

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

        model = pce.Expansion(coef,poly)
        print(resp)
        return poly,coef,model