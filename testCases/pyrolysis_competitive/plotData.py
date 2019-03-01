import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import json

import pyBIT
from pyrolysis_general.src.pyrolysis import PyrolysisCompetitive
from pyrolysis_general.src.read_experiments import ReadExperiments

""" Plot the deterministic data"""

# Write parameters names and initial values
param_names = ["A1", "A2", "A3", "A4", "E1", "E2", "E3", "E4", "g1", "g2",  "g3", "g4"]
param_values = [1.37706107e+01,4.72181620e+00,1.91680940e+00,3.05557096e+01,3.43620979e+04,1.29399652e+05,5.31991965e+04,3.48290136e+04,1.44583100e+00,1.94571847e+00,3.59052337e+00,1.29400649e+00]

# initial 
#param_values =[4.097387026829339, 18.481205222217397,   0.302859645878501,   1.5002610140384331, 34111.7311124222,  128183.83287506446,  51997.65825648188,  34320.30286288397, 6.937260114092008e-05,  0.0036623826495427885 , 0.16429389050789625,  0.2776036939043133]

pyBIT.Metropolis_Hastings_Inference.write_tmp_input_file("reaction_scheme.json", param_names, param_values)
		
# Plot pyro 
pyro_model = PyrolysisCompetitive()	
		
# Read the parameters from the temporary file 
pyro_model.react_reader("tmp_reaction_scheme.json")
pyro_model.param_reader("tmp_reaction_scheme.json")

# Solve the system 
rates = [366, 10]
filenames = ['Bessire_366Kmin.csv', "Wong_10Kmin.csv"]
for rate,file in zip(rates,filenames):
    filename = file
    experiment = ReadExperiments(filename=filename, folder='.')
    numpoints = len(experiment.temperature.values)

    pyro_model.solve_system(temp_0=experiment.temperature.values[0], temp_end=experiment.temperature.values[-1], time=experiment.time.values, beta=rate, n_points=numpoints)

    drho = pyro_model.get_drho_solid()
    T = pyro_model.get_temperature()
    plt.plot(T, experiment.dRho, 'o', mfc='none')
    plt.plot(T, drho)
	
plt.show()

	

	

