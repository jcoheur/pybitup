import spring_model 
import pybitup

case_name = "spring_model"
#case_name = "spring_model_1param" # Uncomment this line to run the case with only one parameter 
input_file_name = "{}.json".format(case_name) 

# Define the model 
my_spring_model = {} 
my_spring_model[case_name] = spring_model.SpringModel(name=case_name)

# Sample 
post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(my_spring_model)
post_dist.__del__()

# Post process 
pybitup.post_process.post_process_data(input_file_name)
