import heat_conduction 
import pyBIT

case_name = "heat_conduction"
input_file_name = "{}.json".format(case_name) 


heat_conduction_model = heat_conduction.HeatConduction()
my_model = pyBIT.Metropolis_Hastings_Inference.Model(heat_conduction_model, heat_conduction_model.compute_temperature, name = case_name)

pyBIT.run_inference.run_inference(input_file_name, my_model)
pyBIT.postProcessData.post_process_data(input_file_name, case_name)
