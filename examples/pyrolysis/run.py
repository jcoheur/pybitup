import sys
sys.path.append('../../')

import one_reaction_pyrolysis
import numpy as np
import pybitup
import matplotlib.pyplot as plt

case_name = "one_reaction_pyrolysis"
input_file_name = "{}.json".format(case_name) 

pyro_model = {}
pyro_model["one_reaction_pyrolysis"] = one_reaction_pyrolysis.OneReactionPyrolysis()



post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(pyro_model)
post_dist.__del__()

pyro_model["366_K_per_Min"] = one_reaction_pyrolysis.OneReactionPyrolysis()

post_dist = pybitup.solve_problem.Propagation(input_file_name)
post_dist.propagate(pyro_model)

post_dist.__del__()

pybitup.post_process.post_process_data(input_file_name)


