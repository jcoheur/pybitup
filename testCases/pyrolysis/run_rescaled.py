import one_reaction_pyrolysis 
import numpy as np
import pyBIT
import matplotlib.pyplot as plt

case_name = "one_reaction_pyrolysis_rescaled"
input_file_name = "{}.json".format(case_name) 

pyro_model = one_reaction_pyrolysis.OneReactionPyrolysis()
parametrization_param = np.array([0, 113000, 2.0, 0.04])

"""
param_values = np.array([1.6635e4, 113000, 2.0, 0.0,  0.04, 6.1])
T = np.array([np.linspace(300.0, 1400.0, 100)])
pyro_model.x = T
pyro_model.param = param_values
gasProd = pyro_model.get_gas_prod()
plt.plot(T[0,:], gasProd[0,:])  

param_values= np.array([1.6635e4, 110000, 2.0, 0.0,  0.04, 6.1])
pyro_model.param = param_values
gasProd = pyro_model.get_gas_prod()
plt.plot(T[0,:], gasProd[0,:])  
plt.show()


# Check that parametrization is correct
parametrization_param = np.array([0, 113000, 2.0, 0.04])
param_values = np.array([1.6635e4, 113000, 2.0, 0.04])
Y = pyro_model.parametrization_forward(param_values, parametrization_param)
X = pyro_model.parametrization_backward(Y, parametrization_param)
# X must be equal to param_values
print(X) 
"""
my_model = pyBIT.Metropolis_Hastings_Inference.Model(pyro_model, pyro_model.compute_gas_prod, pyro_model.parametrization_forward, 
pyro_model.parametrization_backward, pyro_model.det_jac, parametrization_param, case_name)

pyBIT.run_inference.run_inference(input_file_name, my_model)
pyBIT.postProcessData.post_process_data(input_file_name, case_name)