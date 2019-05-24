# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 

import parallel_reaction

# Packages for stochastic inference
import pybit

case_name = "pyrolysis_parallel_1param"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = parallel_reaction.SetParallelReaction()


post_dist = pybit.sample_dist.SolveProblem(input_file_name)
post_dist.sample(pyrolysis_model)
post_dist.post_process_dist()
