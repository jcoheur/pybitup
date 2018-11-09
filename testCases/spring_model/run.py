import spring_model 
import pyBIT

case_name = "spring_model_1param"
input_file_name = "{}.json".format(case_name) 

spring_model = pyBIT.Metropolis_Hastings_Inference.Model(spring_model.model_def, case_name)

pyBIT.run_inference.run_inference(input_file_name, spring_model)
pyBIT.postProcessData.post_process_data(input_file_name, case_name)
