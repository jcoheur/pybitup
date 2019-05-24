import heat_conduction 
import pybit

case_name = "heat_conduction"
input_file_name = "{}.json".format(case_name) 


heat_conduction_model = heat_conduction.HeatConduction()

post_dist = pybit.sample_dist.SolveProblem(input_file_name)
post_dist.sample(heat_conduction_model)
post_dist.post_process_dist()
