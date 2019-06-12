import numpy as np 
from scipy import stats
import random

import pybit.distributions

class BayesianPosterior(pybit.distributions.ProbabilityDistribution):  

    def __init__(self, prior, likelihood, class_model, param_init): 

        self.param_init = param_init
        self.prior = prior
        self.likelihood = likelihood 
        self.model = class_model

        self.dim = len(param_init)
        # for the verification of ito-sde for the 1param (E) pyro model
        #self.distr_support = np.array([[self.model.parametrization_forward(np.array([1e5])), self.model.parametrization_forward(np.array([2e5]))]])
        #self.distr_support = np.array([[np.array([4]), np.array([15])], [np.array([11.3]), np.array([12.95])]])
       
    def get_dim(self): 
        return len(self.param_init)

    def compute_value(self, Y):

        X = self.model.parametrization_backward(Y) 
        
        bayes_post = self.prior.compute_value(X) * self.model.parametrization_det_jac(Y) * self.likelihood.compute_value(X) 

        return bayes_post 

    def compute_log_value(self, Y): 

        X = self.model.parametrization_backward(Y) 

        log_bayes_post = self.prior.compute_log_value(X) + np.log(self.model.parametrization_det_jac(Y)) + self.likelihood.compute_log_value(X)
        
        return log_bayes_post

    def update_eval(self): 

        self.likelihood.update_eval() 

    def save_value(self, name_file): 

        self.likelihood.write_fun_eval(name_file)

    def save_sample(self, fileID_sample, value):

        X = self.model.parametrization_backward(value) 
        fileID_sample.write("{}\n".format(str(X).replace('\n', '')))


class Data:
    """Class defining the data in the inference problem and contains all the information"""

    """     def __init__(self, name="", x=np.array([]), y=np.array([]), n_data_set=0, std_y=np.array([]), index_data_set=np.array([[]])):   
        self.name = list([name])
        self.x = x
        self.num_points = np.array([len(self.x)])
        self.y = y
        self.n_data_set = n_data_set
        self.std_y = std_y
        self.index_data_set = index_data_set """


    def __init__(self, name="", x=np.array([1]), y=np.array([1]), std_y=np.array([1])):
        self.name=list([name])
        self.x=x
        self.num_points=np.array([len(self.x)])
        self.y=y
        self.n_data_set=1
        self.std_y=std_y
        self.index_data_set=np.array([[0,len(self.x)-1]])


    def size_x(self, i):
        """Return the length of the i-th data x"""
        return self.num_points[i]

    def add_data_set(self, new_name, new_x, new_y, new_std_y):
        """Add a set of data to the object"""
        self.name.append(new_name)
        self.x = np.concatenate((self.x, new_x))
        self.num_points = np.concatenate(
            (self.num_points, np.array([len(new_x)])), axis=0)
        self.y = np.concatenate((self.y, new_y))
        self.std_y = np.concatenate((self.std_y, new_std_y))

        # Add new indices array
        last_index = self.index_data_set[self.n_data_set-1, 1]+1
        self.index_data_set = np.concatenate((self.index_data_set, np.array(
            [[last_index, last_index+len(new_x)-1]])), axis=0)

        # Increase the number of dataset
        self.n_data_set += 1


class Model:
    """ Class defining the model function and its reparametrization if specified."""

    def __init__(self, x=[], param=[], scaling_factors_parametrization=1, name=""):
        self.name = name
        self.P = scaling_factors_parametrization
        
        # Variables
        self._x = x

        # Parameters
        self._param = param

    def _get_param(self):
        """Method that is called when we want to read the attribute 'param' """

        return self._param

    def _set_param(self, new_param):
        """Method that is called when we want to modify the attribute 'param' """

        self._param = new_param

    param = property(_get_param, _set_param)

    def _get_x(self):
        """Method that is called when we want to read the attribute 'x' """

        return self._x

    def _set_x(self, new_x):
        """Method that is called when we want to modify the attribute 'x' """

        self._x = new_x

    x = property(_get_x, _set_x)


    def fun_x(self, val_x, val_param): 

        return 1

    def parametrization_forward(self, X=1, P=1):
        
        Y = X

        return Y  

    def parametrization_backward(self, Y=1, P=1):
        
        X = Y

        return X

    def parametrization_det_jac(self, X=1):

        det_jac = 1 

        return det_jac



class Likelihood: 
    """"Class defining the function and the properties of the likelihood function."""

    def __init__(self, exp_data, model_fun): 
        self.data = exp_data
        self.model_fun = model_fun
        self.model_eval = 0

    def compute_value(self, X):
        """ Compute value of the likelihood at the current point X.
        Only gaussian likelihood so far."""

        self.model_eval_X = self.model_fun(X)
        like_val = np.exp(self.sum_of_square(self.data, self.model_eval_X)) 

        return like_val

    def compute_log_value(self, X): 
        
        self.model_eval_X = self.model_fun(X)
        log_like_val = self.sum_of_square(self.data, self.model_eval_X)

        return log_like_val

    def update_eval(self): 

        # We want to save model evaluation for time saving 
        self.model_eval = self.model_eval_X 

    def write_fun_eval(self, name_file):
        """ Save the function evaluation at every save_freq evaluation. """ 

        # We want to save model evaluation for time saving 
        np.save(name_file, self.model_eval)
            

    def compute_ratio(self): 
        """ Compute the ratio of likelihood function"""
        
    def sum_of_square(self, data1, data2):
        """ Compute the sum of square used in the likelihood function.
        data1 is a Data class. 
        data2 is a numpy array. """

        J = - 1/2 * np.sum(((data1.y-data2)/data1.std_y)**2, axis=0)

        return J


def generate_synthetic_data(my_model, std_y, type_pert):
    """ Generate synthetic data based on the model provided in my_model
    with a given standard deviation std_y """

    if type_pert == 'param':
        print("Generate synthetic data based on perturbed parameters")
        num_param = len(my_model.param[:])
        rn_param = np.zeros((1, num_param))
        for i in range(0, num_param):
            rn_param[0, i] = random.gauss(0, std_y)

        my_model.param = my_model.param+rn_param[0, :]
        y = my_model.fun_x()
    else:
        y = my_model.fun_x()

    if type_pert == 'data':
        print("Generate synthetic data based on perturbed nominal solution")
        num_data = len(my_model.x[:])
        rn_data = np.zeros(num_data)

        for i in range(0, num_data):
            rn_data[i] = random.gauss(0, std_y)

        y = y + rn_data[:]

    return y

def write_tmp_input_file(input_file_name, name_param, value_param): 

    # Check inputs
    n_param = len(name_param)
    if n_param is not len(value_param):
        raise ValueError("Parameter names and values must be of the same length") 

    # Open the input file from which we read the data 
    with open(input_file_name) as json_file:

        # Create the new file where we replace the uncertain variables by their values
        with open("tmp_" + input_file_name, "w") as new_input_file: 
        
            # Read json file line by line
            for num_line, line in enumerate(json_file.readlines()):
            
                ind_1 = is_param = line.find("$")
                l_line = len(line)
                while is_param >= 0:		
                
                    ind_2 = ind_1 + line[ind_1+1:l_line].find("$") + 1
                    if ind_2 - ind_1 <= 1: 
                        raise ValueError("No parameter name specified in {} line {} \n{}".format(input_file_name, num_line, line))
                        
                    file_param_name = line[ind_1:ind_2+1]

                    # Temporary variables (see later)
                    new_name_param = list(name_param)
                    new_value_param = list(value_param)	

                    # Check which uncertain parameters are in the current line 
                    for idx, name in enumerate(name_param):
                    
                        key_name = "$" + name + "$"

                        # If the parameter name is in the current line, replace by its value 
                        if file_param_name == key_name: 
                            
                            # Create the new line and unpdate length
                            line = line[0:ind_1-1] + "{}".format(value_param[idx]) + line[ind_2+2:len(line)]
                            l_line = len(line)
                            
                            # Once a parameter name has been found, we don't need to keep tracking it
                            # in the remaining lines
                            new_name_param.remove(name)
                            new_value_param.remove(value_param[idx])
                            
                            # Update index 
                            ind_1 = is_param = line.find("$")
                            
                            
                            break
                        elif idx < n_param-1: 	
                            continue # We go to the next parameter in the list 
                        else: # There is the keyword "$" in the line but the parameter is not in the list 
                            raise ValueError("There is an extra parameter in the line \n ""{}"" " 
                            "but {} is not found in the list.".format(line, line[ind_1+1:ind_2]))
                    
                    if n_param == 0: 
                        # We identified all parameters but we found a "$" in the remaining of the input
                        raise ValueError("We identified all parameters but there is an extra parameter in the line \n ""{}"" " 
                        "but {} is not found in the list.".format(line, line[ind_1+1:ind_2]))
                    
                    # Update parameter lists with only the ones that we still didn't find 
                    name_param = new_name_param
                    value_param = new_value_param
                    n_param = len(name_param)
                    
                    
                # Write the new line in the input file 
                new_input_file.write(line)
                    
    # Check that all params have been found in the input file 
    if len(name_param) > 0:
        raise ValueError("Parameter(s) {} not found in {}".format(name_param, input_file_name)) 