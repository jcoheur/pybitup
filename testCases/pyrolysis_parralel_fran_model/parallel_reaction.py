import numpy as np
import math	

# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 


# Packages for stochastic inference
from pyBIT import Metropolis_Hastings_Inference as MH

# Python packages 


class SetParallelReaction(MH.ModelInference): 
    """ Define the class for the competitive reaction used for the stochastic inference. 
	It calls the model implemented in Francisco's pyrolysis-general toolbox. 
    """
	
    def __init__(self, x=[], param=[]): 
			
        # Initialize parent object ModelInference
        MH.ModelInference.__init__(self)
				
		
    def set_param_values(self, input_file_name, param_names, param_values):
        """Set parameters. For competitive pyrolysis, reactions parameters are read from input file. 
        Uncertain parameters and their values are specified."""
		
		
        # Write the input file.
        MH.write_tmp_input_file(input_file_name, param_names, param_values)

		# Parameters
        self.tau = self._param[0]
		
        # self.time = self._x
        # self.T_0 = self._param[1] 
        # self.T = self.T_0 + self.time * self.tau
        # self.T_end = self.T[-1]
		
        self.T = self._x 
        self.T_0 = self._x[0]
        self.time = (self.T - self.T_0)/(self.tau/60)
        self.T_end = self._x[-1]
		
        self.n_T_steps = len(self._x)

		# Initialize pyrolysis model 
        self.pyro_model = PyrolysisParallel(temp_0=self.T_0, temp_end=self.T_end, time=self.time, beta=self.tau, n_points=self.n_T_steps)
		
		# Read the parameters from the temporary file 
        self.pyro_model.react_reader("tmp_"+input_file_name)
        self.pyro_model.param_reader("tmp_"+input_file_name)
		
		
    def solve_system(self, input_file_name, param_names, param_values): 
		
        # Set parameter
        self.set_param_values(input_file_name, param_names, param_values) 

        # Solve the system  
        self.pyro_model.solve_system()
        #self.pyro_model.compute_analytical_solution()

    def compute_output(self, input_file_name, param_names, param_values):
		
        # Solve the system to get xi_T

        self.solve_system(input_file_name, param_names, param_values)

        return self.pyro_model.get_drho_solid() 
	