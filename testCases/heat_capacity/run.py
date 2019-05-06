import heat_capacity 
import pybit

case_name = "heat_capacity"
input_file_name = "{}.json".format(case_name) 

heat_capacity_model = heat_capacity.HeatCapacity()

post_param_pdf = pybit.inference_problem.Posterior(input_file_name, heat_capacity_model)
post_param_pdf.run_inference()
pybit.post_process.post_process_data(input_file_name)
