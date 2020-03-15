import pybitup.chaoslib as cl
from jsmin import jsmin
from json import loads
import numpy as np
import os.path
import sys

# %% Polynomial Chaos Expansion

class PCE:
    """Wrapper for chaoslib, the class takes a json file containing the polynomial expansion parameters"""

    def __init__(self,parameters): self.param = parameters

    def compute_polynomials(self,point,weight):

        method = self.param["polynomials"]["method"]
        order = self.param["polynomials"]["order"]
        trunc = self.param["polynomials"]["hyperbolic_truncation"]

        if method=="gram_schmidt": return cl.gschmidt(order,point,weight,trunc)
        elif method=="recurrence":

            lawList = []
            for law in self.param["polynomials"]["parameter_laws"]:

                name = list(law.keys())[0]
                param = law[name]

                if name=="uniform": lawList.append(cl.Uniform(*param))
                elif name=="normal": lawList.append(cl.Normal(*param))
                elif name=="gamma": lawList.append(cl.Gamma(*param))
                elif name=="beta": lawList.append(cl.Beta(*param))
                elif name=="expo": lawList.append(cl.Expo(*param))
                elif name=="lognorm": lawList.append(cl.Lognorm(*param))
                else: raise Exception("compute_polynomials: unknown law")
                
            return cl.polyrecur(order,lawList,trunc)

        else: raise Exception("compute_polynomials: unknown method")

    def select_quadrature(self,point,poly):

        method = self.param["quadrature"]["method"]
    
        if method=="fekete": return cl.fekquad(point,poly)
        elif method=="leja": return cl.lejquad(point,poly)
        elif method=="simplex": return cl.linquad(point,poly)
        elif method=="null space": return cl.nulquad(point,poly)
        else: raise Exception("select_quadrature: unknown method")

    def compute_coefficients(self,resp,poly,point,weight):

        method = self.param["coefficients"]["method"]

        if method=="lars" or method=="lars_full":

            it = self.param["coefficients"]["iterations"]
            if it=="unlimited": it = np.inf
            return cl.lars(resp,poly,point,it)

        elif method=="spectral": return cl.spectral(resp,poly,point,weight)
        elif method=="colloc": return cl.colloc(resp,poly,point,weight)
        else: raise Exception("compute_coefficients: unknown method")

    def save_pickle(self,item,name): cl.save(item,name)

    def function_evaluator(self,function,point):

        point = np.atleast_2d(np.transpose(point)).T
        nbrPts = point.shape[0]
        resp = []

        for i in range(nbrPts):

            function.run_model(point[i])
            resp.append(function.model_eval)

        return np.array(resp)

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

        # Computes the polynomials

        poly = self.compute_polynomials(point,weight)

        # Computes the quadrature points and the weights

        if self.param["quadrature"]["method"]=="monte_carlo": weight = None
        elif self.param["quadrature"]["method"]=="quasi_monte_carlo":

            if self.param["quadrature"]["weight_function"]=="None": pdf = None
            else: pdf = eval(self.param["quadrature"]["weight_function_name"])
            nbrPts = int(self.param["quadrature"]["number_points"])
            dom = self.param["quadrature"]["domain"]
            point,weight = cl.qmcquad(nbrPts,dom,pdf)

        else:
            index,weight = self.select_quadrature(point,poly)
            poly.trunc(self.param["quadrature"]["order_truncation"])
            point = point[index]

        # Computes the response at the points

        resp = self.function_evaluator(function,point)

        # Computes the coefficients

        coef = self.compute_coefficients(resp,poly,point,weight)

        if isinstance(coef,tuple):

            index = coef[1]
            coef = coef[0]
            poly.clean(index)
            coef = coef[index]

        if self.param["coefficients"]["method"]=="lars_full": coef = cl.colloc(resp,poly,point)

        # Computes the pce model

        model = cl.Expansion(coef,poly)
        return poly,coef,model