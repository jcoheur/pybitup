import spring_model 
import pybitup

#case_name = "spring_model"
case_name = "spring_model"
input_file_name = "{}.json".format(case_name) 

spring_model = spring_model.SpringModel(name=case_name)

post_dist = pybitup.sample_dist.SolveProblem(input_file_name)
post_dist.sample(spring_model)
post_dist.post_process_dist()
