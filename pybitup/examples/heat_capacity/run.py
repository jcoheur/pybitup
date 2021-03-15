import heat_capacity 
import pybitup
import random 

# Control the seed 
random.seed(a=0)

case_name = "heat_capacity"
input_file_name = "{}.json".format(case_name) 

# Specify the model 
heat_capacity_model = {}
heat_capacity_model[case_name] = heat_capacity.HeatCapacity()

# Sampling 
post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(heat_capacity_model)
post_dist.__del__()

# Post process
pybitup.post_process.post_process_data(input_file_name)

