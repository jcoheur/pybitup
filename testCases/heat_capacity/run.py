import heat_capacity 
import pyBIT

case_name = "heat_capacity"
input_file_name = "{}.json".format(case_name) 

heat_capacity_model = pyBIT.Metropolis_Hastings_Inference.Model(heat_capacity.model_def, case_name)

pyBIT.run_inference.run_inference(input_file_name, heat_capacity_model)
pyBIT.postProcessData.post_process_data(input_file_name, case_name)