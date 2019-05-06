# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 

import parallel_reaction

# Packages for stochastic inference
import pybit

case_name = "pyrolysis_parallel_2param"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = parallel_reaction.SetParallelReaction()
#For ito-sde
#pyrolysis_model = pyBIT.inference_problem.Model(pyro_model, name = case_name)
#for RWMH
#pyrolysis_model = pyBIT.Metropolis_Hastings_Inference.Model(pyro_model, pyro_model.compute_output, name = case_name)

post_param_pdf = pybit.inference_problem.Posterior(input_file_name, pyrolysis_model)
post_param_pdf.run_inference()
pybit.post_process.post_process_data(input_file_name)

