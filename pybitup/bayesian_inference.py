import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt 

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



import pybitup.distributions

class BayesianPosterior(pybitup.distributions.ProbabilityDistribution):  

    def __init__(self, prior, likelihood, class_model, param_init): 

        self.param_init = param_init
        self.prior = prior
        self.likelihood = likelihood 
        self.model = class_model

        self.dim = len(param_init)
        self.name_random_var = self.model.unpar_name 
        
        self.bayes_post = 0
        self.log_bayes_post = 0

    
    def get_dim(self): 
        return len(self.param_init)

    def compute_value(self, Y):

        X = self.model.parametrization_backward(Y) 
           
        bayes_post = self.prior.compute_value(X) * (1/self.model.parametrization_det_jac(X)) * self.likelihood.compute_value(X) 

        self.bayes_post = bayes_post

        return bayes_post 

    def compute_log_value(self, Y): 
        """ Compute the value of the logarithm of the distribution. 
        In the Bayesian framework, this is the posterior distribution, which takes into accout 
        the likelihood, the determinant of the jacobian (if there is a change of variable) and the prior.
        The log(det(jac)) is by default equal to zero if there is no change of variable. """ 

        X = self.model.parametrization_backward(Y) 

        prior_log_value = self.prior.compute_log_value(X)
        log_like_val = self.likelihood.compute_log_value(X)

        if prior_log_value == -np.inf or log_like_val == np.nan:
            # Avoid computation of likelihood if prior is zero
            log_bayes_post = -np.inf
        else: 
            log_bayes_post = prior_log_value - np.log(self.model.parametrization_det_jac(X)) + log_like_val
            # log(1/det_jac) = - log(det_jac)
 
        # Update value
        self.log_bayes_post = log_bayes_post

        return log_bayes_post

    # def compute_log_like(self, Y): 

    #     X = self.model.parametrization_backward(Y)
        
    #     log_like_val = self.likelihood.compute_log_value(X)

    #     return log_like_val

    def compute_grad_log_value(self, Y): 
        """ 
        Compute the gradient of the logarithm of the distribution.
        In the Bayesian framework, this is the log of the posterior 
        distribution, which takes into account the likelihood,
        the determinant of the jacobian and the prior.  

        TODO: 
        !!!! prior still needs to be added here, only constant priors work so far. """

        X = self.model.parametrization_backward(Y)

        grad_log_like = self.likelihood.compute_grad_log_value(X)
        grad_log_det_jac = self.likelihood.compute_grad_log_det_jac(X)

        return grad_log_like - grad_log_det_jac 

    def compute_value_no_reparam(self, X):
        """ For the direct evaluation of the posterior. 
        We do no use the reparameterization in this case. """

        bayes_post = self.prior.compute_value(X) * self.likelihood.compute_value(X) 

        return bayes_post 


    def estimate_hessian_model(self, Y): 
        """ Estimate the Hessian of the model using its analytical gradients. """ 

        X = self.model.parametrization_backward(Y)

        hess_psd = self.likelihood.compute_hess_PSD(X)

        return hess_psd


    def estimate_init_val(self, Y): 
        """ Estimate initial value using optimization algorithm of the local gradient likelihood.""" 
        
        X = self.model.parametrization_backward(Y)

        X_est = self.likelihood.gradient_descent(X, self.likelihood.compute_grad_log_value)

        return self.model.parametrization_forward(X_est)

    def update_eval(self): 

        self.likelihood.update_eval() 

    # def save_log_post(self, IO_fileID): 

    #     IO_fileID['Distribution_values'].write("{}\n".format(str(self.log_bayes_post).replace('\n', '')))

    def save_value(self, IO_fileID, current_it): 

        self.likelihood.write_fun_eval(IO_fileID, current_it)

    def save_sample(self, IO_fileID, value):

        # Write the value in the sampling space 
        IO_fileID['MChains_reparam'].write("{}\n".format(str(value).replace('\n', '')))

        # Write the value in the initial space 
        X = self.model.parametrization_backward(value)
        IO_fileID['MChains'].write("{}\n".format(str(X).replace('\n', '')))

        np.savetxt(IO_fileID['MChains_csv'], np.array([X]), fmt="%f", delimiter=",")


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
        self.dx = np.append(np.diff(self.x), 0.0) 
        self.num_points=np.array([len(self.x)])
        self.y=y
        self.n_runs = len(y.keys()) # Number of runs of the experiment
        self.n_data_set=1
        self.std_y=std_y
        self.index_data_set=np.array([[0,len(self.x)-1]])


        self.mean_y = np.array(self.y[0])
        self.var_s = np.zeros(self.num_points)
        self.std_s = np.array(self.std_y)
        # Estimate sample Statitics if there is more than one experimental run  
        if self.n_runs > 1: 
            # Estimate sample mean 
            for c_run in range(1, self.n_runs): 
                self.mean_y += self.y[c_run]
            self.mean_y = self.mean_y/self.n_runs

            # Estimate sample standard deviation and variance  
            for c_run in range(self.n_runs): 
                self.var_s += (self.y[c_run] - self.mean_y)**2
            self.var_s = self.var_s/(self.n_runs-1)   

            
            self.std_s = np.array(self.var_s**(1/2))
            
            # Use the standard deviation estimated from the provided data set 
            # self.std_y =  self.std_s*np.sqrt(self.n_runs * self.num_points) + 1e-10 # To avoid very small values 
            # Use the standard deviation provided in input file 
            self.std_y =  self.std_y*np.sqrt(self.n_runs * self.num_points) + 1e-10 # To avoid very small values 

            print(self.std_y)

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

class Data_2:

    def __init__(self, x=np.array([1]), y=np.array([1]), std_y=np.array([1])):
        self._x=x
        self._y=x
        self._std_y=std_y

    def _get_x(self):

        return self._x

    def _set_x(self, new_x):

        self._x = new_x

    x = property(_get_x, _set_x)

    def _get_y(self):

        return self._y

    def _set_y(self, new_y):

        self._y = new_y

    y = property(_get_y, _set_y)

    def _get_std_y(self):

        return self._std_y

    def _set_std_y(self, new_std_y):

        self._std_y = new_std_y

    std_y = property(_get_std_y, _set_std_y)

    




class Model:
    """ Class defining the model function and its reparametrization if specified."""

    def __init__(self, x=[], param=[], scaling_factors_parametrization=1, name=""):
        self.name = name
        self.P = scaling_factors_parametrization
        
        # Variables
        self._x = x

        # Parameters
        self._param = param
        self.param_nom = []
        self.param_names = []
        self.input_file_name = [] 
        self.unpar_name = {}
        self.unpar_name_dict = {}

        # Value of the model f(param, x) 
        self.model_eval = 0

        self.model_grad = 0

    def size_x(self):
        """Return the length of the i-th data x"""
        return len(self._x) 

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


    def fun_x(self, *ext_parameters): 

        return 1

    def d_fx_dparam(self, *ext_parameters): 

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

    def parametrization_inv_jac(self, X=1):

        inv_jac = X

        return inv_jac 

    def run_model(self, var_param): 
        """ Define the vector of model evaluation."""
        

        if not self.input_file_name: 
            # In this case, the model parameters are not in an input file. They are all specified 
            # in param_names. Some of them are uncertain (specified in the Prior inputs) and we therefore 
            # need to define which are uncertain and which are not. 

            n_param_model = len(self.param_nom)

            self.param = self.param_nom
        
            var_param_index = []
            char_name = " ".join(self.unpar_name)

            n_unpar = len(self.unpar_name)

            for idx, name in enumerate(self.param_names):
                is_name = char_name.find(name)
                if is_name >= 0: 
                    var_param_index.append(idx)

            if n_unpar < n_param_model: # Only a subset of parameters is uncertain
                vec_param = self.param_nom 
                for n in range(0, n_unpar): 
                    vec_param[var_param_index[n]] = var_param[n]

                self.param = vec_param

            else: # All parameters are uncertain
                self.param = var_param

            self.model_eval = self.fun_x()
            
            
        else: 
            # Model is build based on a given input file. 
            # If an input file is provided, the uncertain parameters are specified within the file. 
            # The uncertain parameters is identified by the keyword "$", e.g. "$myParam$". 
            # The input file read by the model is in user_inputs['Model']['input_file']. 
        
            #model_eval = np.concatenate((model_eval, my_model.fun_x(c_model['input_file'], unpar_name, var_param)))

            self.model_eval = self.fun_x(self.input_file_name, self.unpar_name, var_param)



    def get_gradient_model(self, var_param): 
        """ Evaluate the gradient of the model with respect to its parameter at var_param""" 

        self.model_grad = self.d_fx_dparam()

class Emulator(Model):
    """ Class defining the emulator of a model."""

    def __init__(self, x=[], param=[], scaling_factors_parametrization=1, name=""):
        Model.__init__(self, x=[], param=[], scaling_factors_parametrization=1, name="")



class Likelihood: 
    """"Class defining the function and the properties of the likelihood function."""

    def __init__(self, exp_data, model_list): 
        self.data = exp_data
        self.models = model_list 
        self.SS_X = 0 # sum of square 
        self.arg_LL = 0 # Arg of the likelihood function 

        # self.model_eval[model_id] is the value that is updated and saved
        # We initialise it here as a list  
        self.model_eval = {}


    def compute_value(self, X):
        """ Compute value of the likelihood at the current point X.
        Only gaussian likelihood so far."""

        # self.model_eval_X = self.model_fun(X)
        # like_val = np.exp(self.sum_of_square(self.data, self.model_eval_X)) 

        self.arg_gauss_likelihood(X)
        like_val = np.exp(- (1/2) * self.arg_LL) 

        return like_val

    def compute_log_value(self, X): 
        """ Compute the log of the likelihood function (up to a constant). """

        # self.model_eval_X = self.model_fun(X)
        # log_like_val = self.sum_of_square(self.data, self.model_eval_X)
        
        self.arg_gauss_likelihood(X)
        log_like_val = - (1/2) * self.arg_LL 

        return log_like_val

    def update_eval(self): 
        """ Update the value of the model evaluation. """ 

        # Initialise model evaluation value 
        for model_id in self.models.keys(): 
            self.model_eval[model_id] = []

        # Update it 
        for model_id in self.models.keys(): 
            self.model_eval[model_id] = np.concatenate((self.model_eval[model_id], self.models[model_id].model_eval))

    def write_fun_eval(self, IO_util, num_it):
        """ Save the function evaluation in an output file. 
        The name of the output file is fixed.""" 

        for model_id in self.models.keys(): 
            np.save(IO_util['path']['fun_eval_folder']+'/'+model_id+'_fun_eval.'+str(num_it), self.model_eval[model_id]) 

            # n_x = len(self.data[model_id].x)
            # n_data_set = int(len(self.data[model_id].y)/n_x)

            # for i in range(n_data_set): 
            #     plt.figure(i)
            #     plt.plot(self.data[model_id].x, self.data[model_id].y[i*n_x:(i+1)*n_x], '--')
            #     plt.plot(self.data[model_id].x, self.models[model_id].model_eval[i*n_x:(i+1)*n_x])  

        #plt.show()

    def compute_ratio(self): 
        """ Compute the ratio of likelihood function"""
        
    def sum_of_square(self, X):
        """ Compute the sum of square in the least square sense. 
        X is the value at which we evaluate the model. """

        J = 0 
        for model_id in self.models.keys(): 

            # Compute value for the model at X 
            self.models[model_id].run_model(X)

            # n_x = len(self.data[model_id].x)
            # n_data_set = int(len(self.data[model_id].y)/n_x)

            # for i in range(n_data_set): 
            #     plt.figure(i)
            #     plt.plot(self.data[model_id].x, self.data[model_id].y[i*n_x:(i+1)*n_x], '--')
            #     plt.plot(self.data[model_id].x, self.models[model_id].model_eval[i*n_x:(i+1)*n_x])

            # Compute the sum of square 
            for c_run in range(self.data[model_id].n_runs): 
                arg_exp = (self.data[model_id].y[c_run] - self.models[model_id].model_eval)
                J = J  + arg_exp**2

        #plt.show()
        self.SS_X = J

    def arg_gauss_likelihood(self, X):
        """ Compute the weighted sum of square which is the argument of the gaussian likelihood. """ 

        J = 0 
        J2 = 0
        for model_id in self.models.keys(): 

            # Compute value for the model at X 
            self.models[model_id].run_model(X)

            
            # Compute the weighted sum of square 
            for c_run in range(self.data[model_id].n_runs): 
                # dy = self.data[model_id].y[c_run] - self.models[model_id].model_eval


                # int1 = np.sum(self.models[model_id].model_eval * self.data[model_id].dx/self.data[model_id].std_y)
                # int2 = np.sum(self.data[model_id].y[c_run] * self.data[model_id].dx/self.data[model_id].std_y)
                # frac_y = np.array(int1 / int2)
                # new_std_y = np.array(dy) / 100
                # new_std_y2 = np.array(self.data[model_id].y[0])*1e-3 + 1e-18


                # arg_exp = (self.data[model_id].y[c_run] - self.models[model_id].model_eval)/(new_std_y)
                # arg_exp2 = (self.data[model_id].y[c_run] - self.models[model_id].model_eval)/(new_std_y2)

                #arg_exp = (int1 - int2)
                arg_exp = (self.data[model_id].y[c_run] - self.models[model_id].model_eval)/(self.data[model_id].std_y)
                #arg_exp = arg_exp[9]
                J = J  + np.sum(arg_exp**2, axis=0)

                # plt.figure(1)
                # plt.plot(self.data[model_id].x, new_std_y**2)
                # plt.plot(self.data[model_id].x, new_std_y2**2)
                # plt.figure(2)
                # plt.plot(self.data[model_id].x, arg_exp**2)
                # plt.plot(self.data[model_id].x, arg_exp2**2)
            #print(J)
            # For stochastic processes : compute the difference between the two using integral 
            # for c_run in range(self.data[model_id].n_runs):  
            #     arg_exp = (self.data[model_id].y[c_run] - self.models[model_id].model_eval)/(self.data[model_id].std_y)*self.data[model_id].dx
            #     J2 = J2  + np.sum(arg_exp**2, axis=0)

        # plt.figure(2)

        #plt.show()
        #print(J, J2)
        self.arg_LL = J


    # # Implementation of the 15-01-20 
    # def compute_grad_log_value(self, X): 

    #     grad = np.zeros(len(X))
    #     for model_id in self.models.keys(): 

    #         # Compute gradient of the model 
    #         self.models[model_id].run_model(X)
    #         self.models[model_id].get_gradient_model(X)
    #         inv_jac = self.models[model_id].parametrization_inv_jac(X)

    #         # Compute grad log LL 
    #         ss_x = (self.data[model_id].y - self.models[model_id].model_eval)/(self.data[model_id].std_y**2) 
    #         for i, pn in enumerate(self.models[model_id].unpar_name):

    #             # grad_model_i = self.models[model_id].model_grad[0, pn]
    #             grad_model_i = []
    #             nspecies = 1 #14, 2, 1
    #             for j in range(nspecies): 
    #                 grad_model_i = np.concatenate((grad_model_i, self.models[model_id].model_grad[j, pn]))
                
    #             prod_i = ss_x * grad_model_i * inv_jac[i, i]

    #             grad[i] = np.sum(prod_i, axis=0)

    #     return grad


    def compute_grad_log_value(self, X): 
        """ 
        A function that computes the gradient of the log-likelihood with respect
        to the parameter values containe in X. 
        This is for the Gaussian likelihood, such that the gradient of the log
        is simply the gradient of the argument (the sum of square weighted by
        the sigmas times -1/2).

        This function is called when the analytical gradients are required. 

        Parameters
        ----------
        X: numpy array
            Parameter values   

        Returns
        ---------
        grad: numpy array
            Gradient of the log-likelihood with respect to the different 
            parameters values

        Joffrey Coheur. Last update: 03-12-20. 

        """

        grad = np.zeros(len(X))
        for model_id in self.models.keys(): 

            # Compute gradient of the model 
            self.models[model_id].run_model(X)
            self.models[model_id].get_gradient_model(X)
            inv_jac = self.models[model_id].parametrization_inv_jac(X)

            # TODO : this is not yet updated with c_run !! 
            # Compute grad log LL 
            ss_x = (self.data[model_id].y[0] - self.models[model_id].model_eval)/(self.data[model_id].std_y**2) 

            for i, pn in enumerate(self.models[model_id].unpar_name):

                # grad_model_i = self.models[model_id].model_grad[0, pn]
                # TODO: This still depends on the number of species  !! 
                # nspecies = 1 #14, 2, 1
                # for j in range(nspecies): 
                #     grad_model_i = np.concatenate((grad_model_i, self.models[model_id].model_grad[j, pn]))
                my_mat = np.matmul(ss_x, self.models[model_id].model_grad[pn])  # ss_x * grad_model_i 
                #grad = np.dot(inv_jac, my_mat) 
                for k, pn2 in enumerate(self.models[model_id].unpar_name):
                    #prod_i_k = my_mat * inv_jac[i, k]
                    grad[k] = grad[k] + my_mat * inv_jac[i, k]

        return grad


    def compute_grad_log_det_jac(self, X):

        for model_id in self.models.keys(): 

            inv_jac = self.models[model_id].parametrization_inv_jac(X)
            grad_det_jac = self.models[model_id].grad_det_jac(X)
            det_jac = self.models[model_id].parametrization_det_jac(X)

            grad_log_det_jac = np.matmul(grad_det_jac, inv_jac)/det_jac

        return grad_log_det_jac

    def compute_hess_PSD(self, X): 

        hess_PSD = np.zeros((len(X), len(X)))
        for model_id in self.models.keys(): 

            # Compute gradient of the model 
            self.models[model_id].run_model(X)
            self.models[model_id].get_gradient_model(X)
            inv_jac = self.models[model_id].parametrization_inv_jac(X)

            for i, pn_i in enumerate(self.models[model_id].unpar_name):

                # grad_model_i = self.models[model_id].model_grad[0, pn]
                grad_model_i = []
                nspecies = 14 #14, 2, 1
                for ns in range(nspecies): 
                    grad_model_i = np.concatenate((grad_model_i, self.models[model_id].model_grad[ns, pn_i]))
                prod_i = grad_model_i * inv_jac[i] 

                for j, pn_j in enumerate(self.models[model_id].unpar_name):
                    if j < i: 
                        continue
                    else: 
                        grad_model_j = []
                        for ns in range(nspecies):     
                            grad_model_j = np.concatenate((grad_model_j, self.models[model_id].model_grad[ns, pn_j]))
                        prod_j = grad_model_j * inv_jac[j] 
                    prod_ij = prod_i * prod_j / self.data[model_id].std_y**2 

                    hess_PSD[i, j] = np.sum(prod_ij, axis=0) 
                    hess_PSD[j, i] = hess_PSD[i, j]

        return hess_PSD


        
    def gradient_descent(self, X, grad_X):
        """ First attempt to use optimization algorithm to find an estimate. 
        Gradient descent algorithm. """ 

        X_n = X
        X_n_p = X

        vec_grad = grad_X(X_n)
        gamma = 1

        max_iter = 1000
        for i in range(max_iter):
            
            X_n_p = X_n + gamma *  vec_grad

            # Updates
            X_n = X_n_p
            vec_grad = grad_X(X_n)
            #norm_grad = np.linalg.norm(vec_grad)

        self.arg_gauss_likelihood(X_n)

        return X_n


def generate_synthetic_data(my_model, std_param, std_y):
    """ Generate synthetic data based on the model provided in my_model
    with a given standard deviation std_y """


    # Perturbe nominal parameter values 

    num_param = len(my_model.param[:])
    rn_param = np.zeros((1, num_param))
    for i in range(0, num_param):
        rn_param[0, i] = random.gauss(0, std_param[i])
    my_model.param = my_model.param+rn_param[0, :]

    # Generate solution using pertubrbed parameter values
    y_pert = my_model.fun_x()


    # Add experimental noise to the solution 
    num_data = len(my_model.x[:])
    rn_data = np.zeros(num_data)

    for i in range(0, num_data):
        rn_data[i] = random.gauss(0, std_y[i])
    y_noisy = y_pert + rn_data[:]

    return y_noisy

def write_tmp_input_file(input_file_name, name_param, value_param): 

    # Check inputs
    n_param = len(name_param)
    if n_param is not len(value_param):
        raise ValueError("Parameter names and values must be of the same length") 

    # Open the input file from which we read the data 
    with open(input_file_name) as json_file:

        # Create the new file where we replace the uncertain variables by their values
        with open("tmp_proc_"+str(rank)+'_' + input_file_name, "w") as new_input_file: 
        
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