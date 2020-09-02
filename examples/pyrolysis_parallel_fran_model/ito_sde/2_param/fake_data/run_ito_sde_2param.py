# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 

import parallel_reaction_2param

# Packages for stochastic inference
import pybitup

case_name = "pyrolysis_parallel_2param"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = {}
pyrolysis_model["case_name"] = parallel_reaction_2param.SetParallelReaction()

post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(pyrolysis_model)
post_dist.__del__()

pybitup.post_process.post_process_data(input_file_name)
