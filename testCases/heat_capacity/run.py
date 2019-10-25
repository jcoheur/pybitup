import heat_capacity 
import pybitup

case_name = "heat_capacity"
input_file_name = "{}.json".format(case_name) 

heat_capacity_model = heat_capacity.HeatCapacity()

post_dist = pybitup.sample_dist.SolveProblem(input_file_name)
post_dist.sample(heat_capacity_model)
post_dist.post_process_dist()

