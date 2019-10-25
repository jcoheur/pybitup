# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisParallel 

import parallel_reaction

# Packages for stochastic inference
import pybitup
import matplotlib.pyplot as plt

case_name = "pyrolysis_parallel"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = parallel_reaction.SetParallelReaction()

post_dist = pybitup.sample_dist.SolveProblem(input_file_name)
post_dist.sample(pyrolysis_model)
post_dist.post_process_dist()



