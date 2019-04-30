# Packages for the pyrolysis model
from src.pyrolysis import PyrolysisParallel 

import parallel_reaction

# Packages for stochastic inference
import pyBIT
import matplotlib.pyplot as plt

# Python packages
from scipy import linalg, stats


case_name = "pyrolysis_parallel_2param"
input_file_name = "{}.json".format(case_name) 

pyro_model = parallel_reaction.SetParallelReaction()
#For ito-sde
pyrolysis_model = pyBIT.Metropolis_Hastings_Inference.Model(pyro_model, pyro_model.compute_output, function_f = pyro_model.cv_forward, 
function_b = pyro_model.cv_backward, name = case_name)
#for RWMH
#pyrolysis_model = pyBIT.Metropolis_Hastings_Inference.Model(pyro_model, pyro_model.compute_output, name = case_name)

pyBIT.run_inference.run_inference(input_file_name, pyrolysis_model)
pyBIT.postProcessData.post_process_data(input_file_name)

