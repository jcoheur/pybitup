import heat_conduction 
import pybit

case_name = "heat_conduction"
input_file_name = "{}.json".format(case_name) 


heat_conduction_model = heat_conduction.HeatConduction()

post_param_pdf = pybit.inference_problem.Posterior(input_file_name, heat_conduction_model)
post_param_pdf.run_inference()
pybit.post_process.post_process_data(input_file_name)
