import heat_conduction 
import pybitup

case_name = "heat_conduction"
input_file_name = "{}.json".format(case_name) 

heat_conduction_model = {}
heat_conduction_model[case_name] = heat_conduction.HeatConduction()

post_dist = pybitup.sample_dist.SolveProblem(input_file_name)
post_dist.sample(heat_conduction_model)
post_dist.post_process_dist()
