import one_reaction_pyrolysis 
import numpy as np
import pyBIT
import matplotlib.pyplot as plt

case_name = "one_reaction_pyrolysis"
input_file_name = "{}.json".format(case_name) 

pyro_model = one_reaction_pyrolysis.OneReactionPyrolysis()

#param_values = np.array([1.6635e4, 113000, 2.0, 0.0,  0.04, 6.1])
#T = np.array([np.linspace(300.0, 1400.0, 100)])
#gasProd = one_reaction_pyrolysis.model_def(T, param_values)
#print(gasProd[0,:])
#plt.plot(T[0,:], gasProd[0,:])  
#plt.show()

# Check that parametrization is correct
#parametrization_param = np.array([0, 1.4e5, 3.5, 2.6e-4])
#param_values = np.array([1.6635e4, 113000, 2.0, 0.04])
#Y = one_reaction_pyrolysis.parametrization_forward(param_values, parametrization_param)
#X = one_reaction_pyrolysis.parametrization_backward(Y, parametrization_param)
# X must be equal to param_values
# print(X) 

my_model = pyBIT.Metropolis_Hastings_Inference.Model(pyro_model, pyro_model.compute_gas_prod, name = case_name)

pyBIT.run_inference.run_inference(input_file_name, my_model)
pyBIT.postProcessData.post_process_data(input_file_name)
