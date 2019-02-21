import spring_model 
import pyBIT

#case_name = "spring_model"
case_name = "spring_model_1param"
input_file_name = "{}.json".format(case_name) 

spring_model = spring_model.SpringModel()
my_model = pyBIT.Metropolis_Hastings_Inference.Model(spring_model, spring_model.compute_elongation, name = case_name)

pyBIT.run_inference.run_inference(input_file_name, my_model)
pyBIT.postProcessData.post_process_data(input_file_name, case_name)

