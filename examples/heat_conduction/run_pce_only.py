import sys
sys.path.append('../../')

import heat_conduction
import pybitup

case_name = "heat_conduction_pce_only"
input_file_name = "{}.json".format(case_name) 

heat_conduction_model = {}
heat_conduction_model[case_name] = heat_conduction.HeatConduction()

post_dist = pybitup.solve_problem.Propagation(input_file_name)
post_dist.propagate(heat_conduction_model)
post_dist.__del__()
