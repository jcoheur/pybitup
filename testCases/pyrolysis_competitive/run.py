# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisCompetitive 
from pyrolysis_general.src.read_experiments import ReadExperiments
import competitive_reaction

# Packages for stochastic inference
import pybit
import matplotlib.pyplot as plt

# Python packages
from scipy import linalg, stats


case_name = "pyrolysis_competitive"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = competitive_reaction.SetCompetitiveReaction()

post_param_pdf = pybit.inference_problem.Posterior(input_file_name, pyrolysis_model)
post_param_pdf.run_inference()
pybit.post_process.post_process_data(input_file_name)


