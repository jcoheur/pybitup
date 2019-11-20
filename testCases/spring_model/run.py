import spring_model 
import pybitup

#case_name = "spring_model"
case_name = "spring_model"
input_file_name = "{}.json".format(case_name) 

my_spring_model = {} 
my_spring_model[case_name] = spring_model.SpringModel(name=case_name)

post_dist = pybitup.sample_dist.SolveProblem(input_file_name)
post_dist.sample(my_spring_model)
post_dist.__del__()

pybitup.post_process.post_process_data(input_file_name)
