import spring_model 
import pybit

#case_name = "spring_model"
case_name = "spring_model_1param"
input_file_name = "{}.json".format(case_name) 

spring_model = spring_model.SpringModel(name=case_name)

post_param_pdf = pybit.inference_problem.Posterior(input_file_name, spring_model)
post_param_pdf.run_inference()
pybit.post_process.post_process_data(input_file_name)
