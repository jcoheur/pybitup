import sys
sys.path.append('../../')

import one_reaction_pyrolysis

import numpy as np
import pybitup
import matplotlib.pyplot as plt

case_name = "one_reaction_pyrolysis"
input_file_name = "{}.json".format(case_name) 

# Sampling using pyrolysis production data 
# Set the model 
pyro_model = {}
pyro_model["one_reaction_pyrolysis"]  = one_reaction_pyrolysis.OneReactionPyrolysis()
# Sample 
post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(pyro_model)
post_dist.__del__()

pybitup.post_process.post_process_data(input_file_name)

