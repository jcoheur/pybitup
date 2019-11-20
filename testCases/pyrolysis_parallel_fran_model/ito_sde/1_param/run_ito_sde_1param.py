# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 

import parallel_reaction_1param

# Packages for stochastic inference
import pybitup

case_name = "pyrolysis_parallel_1param"

input_file_name = "{}.json".format(case_name) 

pyrolysis_model = {}
pyrolysis_model["parallel_pyrolysis_1param"] = parallel_reaction_1param.SetParallelReaction()


post_dist = pybitup.sample_dist.SolveProblem(input_file_name)
post_dist.sample(pyrolysis_model)
post_dist.post_process_dist()
