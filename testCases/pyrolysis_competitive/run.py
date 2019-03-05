# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisCompetitive 
from pyrolysis_general.src.read_experiments import ReadExperiments
import competitive_reaction

# Packages for stochastic inference
import pyBIT
import matplotlib.pyplot as plt

# Python packages
from scipy import linalg, stats


case_name = "pyrolysis_competitive"
input_file_name = "{}.json".format(case_name) 

pyro_model = competitive_reaction.SetCompetitiveReaction()

pyrolysis_model = pyBIT.Metropolis_Hastings_Inference.Model(pyro_model, pyro_model.compute_output, name = case_name)


pyBIT.run_inference.run_inference(input_file_name, pyrolysis_model)
pyBIT.postProcessData.post_process_data(input_file_name)

