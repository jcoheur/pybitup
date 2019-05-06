# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 

import parallel_reaction

# Packages for stochastic inference
import pybit
import matplotlib.pyplot as plt

# Python packages
from scipy import linalg, stats


case_name = "pyrolysis_parallel"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = parallel_reaction.SetParallelReaction()

post_param_pdf = pybit.inference_problem.Posterior(input_file_name, pyrolysis_model)
post_param_pdf.run_inference()
pybit.post_process.post_process_data(input_file_name)


