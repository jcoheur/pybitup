import heat_capacity 
import pybit

case_name = "heat_capacity"
input_file_name = "{}.json".format(case_name) 

heat_capacity_model = heat_capacity.HeatCapacity()

post_dist = pybit.sample_dist.SolveProblem(input_file_name)
post_dist.sample(heat_capacity_model)
post_dist.post_process_dist()

