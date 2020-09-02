import numpy as np
import math	

# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 

# Packages for stochastic inference
from pybitup import bayesian_inference as bi

# Python packages 


class SetParallelReaction(bi.Model): 
    """ Define the class for the competitive reaction used for the stochastic inference. 
    	It calls the model implemented in Francisco's pyrolysis-general toolbox. 
    """
    	
    def __init__(self, x=[], param=[], scaling_factors_parametrization=1, name=""):
        bi.Model.__init__(self,x, param, scaling_factors_parametrization, name)

        R = 8.314
        self.P = np.array([1.0, 92000, 2.5, 0.1, 1.0, 122000, 1.3, 0.05])
        self.X1 = self.P[1] / (R * 686)
        self.X2 = self.P[5] / (R * 909)
    		
    def set_param_values(self, input_file_name, param_names, param_values):
        """Set parameters. For competitive pyrolysis, reactions parameters are read from input file. 
        Uncertain parameters and their values are specified."""
        
        # Write the input file.
        bi.write_tmp_input_file(input_file_name, param_names, param_values)
        
        # Parameters
        self.tau = self.param[0]
        				
        # Variables
        self.T = self.x 
        self.T_0 = self._x[0]
        self.time = (self.T - self.T_0)/(self.tau/60)
        self.T_end = self.x[-1]
        		
        self.n_T_steps = len(self.x)
        
        		# Initialize pyrolysis model 
        self.pyro_model = PyrolysisParallel(temp_0=self.T_0, temp_end=self.T_end, time=self.time, beta=self.tau, n_points=self.n_T_steps)
        		
        		# Read the parameters from the temporary file 
        self.pyro_model.react_reader("tmp_proc_0_"+input_file_name)
        self.pyro_model.param_reader("tmp_proc_0_"+input_file_name)		
    		
    def solve_system(self, input_file_name, param_names, param_values): 
    		
        # Set parameter
        self.set_param_values(input_file_name, param_names, param_values) 
    
        # Solve the system  
        #self.pyro_model.solve_system()
        self.pyro_model.compute_analytical_solution()
    
    
    def fun_x(self, input_file_name, param_names, param_values):
    		
        # Solve the system to get xi_T
    
        self.solve_system(input_file_name, param_names, param_values)
    
        return self.pyro_model.get_drho_solid() 
    
    		
    def parametrization_forward(self, X):

        Y = np.zeros(len(X[:]))
        
        Y[0] = np.log(X[0]) - X[1] / self.P[1] * self.X1
        
        Y[1] =  X[1] / self.P[1]
        
        Y[2] = X[2] / self.P[2]
        
        Y[3] = X[3] / self.P[3]
        		
        Y[4] = np.log(X[4]) - X[5] / self.P[5] * self.X2
        
        Y[5] =  X[5] / self.P[5]
        
        Y[6] = X[6] / self.P[6]
        
        Y[7] = X[7] / self.P[7]
        		
    
        return Y

    def parametrization_backward(self, Y):

        X = np.zeros(len(Y[:]))
        
        X[0] = np.exp(Y[0] + Y[1]*self.X1)
        
        X[1] = Y[1] * self.P[1]
        
        X[2] = Y[2] * self.P[2]
        
        X[3] = Y[3] * self.P[3]
        		
        X[4] = np.exp(Y[4] + Y[5]*self.X2)
        
        X[5] = Y[5] * self.P[5]
        
        X[6] = Y[6] * self.P[6]
        
        X[7] = Y[7] * self.P[7]
        		

        return X
        
    def parametrization_det_jac(self, X):
    
        return 1/(X[0]*X[4]) 

		
   