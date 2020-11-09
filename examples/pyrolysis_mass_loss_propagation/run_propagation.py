import sys
sys.path.append('../../')

import model_mass_loss

import numpy as np
import pybitup
import matplotlib.pyplot as plt

case_name = "propagation_mass_loss"
input_file_name = "{}.json".format(case_name) 

# Set model for the mass loss 
pyro_model = {}
pyro_model["mass_loss"]  = model_mass_loss.MassLoss()

# Propagation for the mass loss
post_dist = pybitup.solve_problem.Propagation(input_file_name)
post_dist.propagate(pyro_model)
post_dist.__del__()

pybitup.post_process.post_process_data(input_file_name)

