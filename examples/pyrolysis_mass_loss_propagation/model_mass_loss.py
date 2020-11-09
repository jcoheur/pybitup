from scipy import special
import numpy as np
import math	

from pybitup import bayesian_inference as bi

from one_reaction_pyrolysis import OneReactionPyrolysis


class MassLoss(OneReactionPyrolysis): 

    R = 8.314

    def __init__(self, x=[], param=[]): 
            
        # Initialize parent object ModelInference
        OneReactionPyrolysis.__init__(self)

    def fun_x(self): 

        # We just need to re-define the fun_x method to return the mass loss for the propagation 

        # Solve the system to get xi_T
        self.solve_system()

        return self.compute_mass_loss()