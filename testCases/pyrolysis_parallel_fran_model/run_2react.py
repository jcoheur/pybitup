# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 

import parallel_reaction

# Packages for stochastic inference
import pybitup


case_name = "pyrolysis_parallel_2react"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = parallel_reaction.SetParallelReaction()

my_dist = pybitup.sample_dist.SolveProblem(input_file_name)
my_dist.sample(pyrolysis_model)
my_dist.post_process_dist()

