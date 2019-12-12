import numpy as np
import random
import scipy
from scipy import linalg
import time
import pandas as pd 

class MetropolisHastings:

    def __init__(self, IO_fileID, caseName, nIterations, param_init, V, prob_distr):
        self.caseName = caseName
        self.nIterations = nIterations
        if self.nIterations < 100:
            self.save_freq = 1
        else:
            self.save_freq = nIterations/100
        self.param_init = param_init
        self.V = V
        self.prob_distr = prob_distr 
        self.distr_fun = prob_distr.compute_log_value

        # Initialize parameters and functions
        self.n_param = self.param_init.size
        self.current_val = self.param_init
        self.new_val = np.zeros(self.n_param)
        self.z_k = np.zeros(self.n_param)
        self.distr_fun_current_val = self.distr_fun(self.current_val)

        # # Initial variance estiamte 
        # prob_distr.likelihood.sum_of_square(self.current_val)
        # for model_id in prob_distr.likelihood.data.keys():
        #     sigma_k_square = prob_distr.likelihood.SS_X / (prob_distr.likelihood.data[model_id].num_points - self.n_param)
        #     prob_distr.likelihood.data[model_id].std_y = np.sqrt(sigma_k_square)


        # Outputs 
        self.IO_fileID = IO_fileID
        # Ensure that we start the files at 0 (due to the double initialisation of DRAM)
        self.IO_fileID['MChains'].seek(0) 
        self.IO_fileID['MChains_reparam'].seek(0)
        self.IO_fileID['MChains_csv'].seek(0)
        self.distr_output_file_name = "output/fun_eval."

        # Write initial values 
        self.prob_distr.save_sample(self.IO_fileID, self.current_val)
        self.prob_distr.update_eval()
        self.write_fun_distr_val(0)

        # Cholesky decomposition of V. Constant if MCMC is not adaptive
        self.V_0 = V 
        self.R = linalg.cholesky(V)

        # Monitoring the chain
        self.n_rejected = 0

        # Print current time and start clock count
        print("Start time {}" .format(time.asctime(time.localtime())))
        self.t1 = time.clock()

        self.it = 0 

        # Monitoring acceptation rate 
        self.mean_r = 0

    def run_algorithm(self):

        for self.it in range(1, self.nIterations+1):
            self.compute_new_val()
            self.compute_acceptance_ratio()
            self.accept_reject()
            # self.update_sigma() 

            # We estimate time and monitor acceptance ratio 
            self.mean_r = self.mean_r + self.r 
            if self.it % (self.nIterations/1000) == 0:
                self.compute_time(self.t1)

            # Write the sample values and function evaluation in a file 
            self.prob_distr.save_sample(self.IO_fileID, self.current_val)
            self.write_fun_distr_val(self.it)


        self.compute_covariance()

        self.terminate_loop()

    def compute_new_val(self):

        # Guess parameter (candidate or proposal)
        self.z_k = self.compute_multivariate_normal()

        # Write the standard gaussian normal proposal value, whatever is was accepted or rejected 
        # self.IO_fileID['gp'].write("{}\n".format(str(self.z_k).replace('\n', '')))

        # Compute new value 
        self.new_val[:] = self.current_val[:] + np.transpose(np.matmul(self.R, np.transpose(self.z_k)))

    def compute_acceptance_ratio(self):

        self.distr_fun_new_val = self.distr_fun(self.new_val[:])
       
        if self.distr_fun_new_val is -np.inf: 
            self.r = 0
        else: 
            #self.r = self.distr_fun_new_val/self.distr_fun_current_val
            self.r = np.exp(self.distr_fun_new_val - self.distr_fun_current_val)

        self.alpha = min(1, self.r)
        
    def accept_reject(self):

        # Update
        u = random.random()  # Uniformly distributed number in the interval [0,1)
        if u < self.alpha:  # Accepted
            self.current_val[:] = self.new_val[:]
            self.distr_fun_current_val = self.distr_fun_new_val
            self.prob_distr.update_eval()
        else:  # Rejected, current val remains the same
            self.n_rejected += 1

    def update_sigma(self): 
        ns = .01
        self.prob_distr.likelihood.sum_of_square(self.current_val)
        for model_id in self.prob_distr.likelihood.data.keys():
            #print(self.prob_distr.likelihood.data[model_id].std_y)
            a = 0.5 * (ns + self.prob_distr.likelihood.data[model_id].num_points) 
            b = 0.5 * (ns * 0.2604**2 +  self.prob_distr.likelihood.SS_X)

            self.prob_distr.likelihood.data[model_id].std_y = np.sqrt(scipy.stats.invgamma.rvs(a, loc=0, scale=b))
            #print(self.prob_distr.likelihood.data[model_id].std_y)

    def compute_covariance(self):

        # Load all the previous iterations
        param_values = np.zeros((self.nIterations+2, self.n_param))
        j = 0
        self.IO_fileID['MChains'].seek(0)
        for line in self.IO_fileID['MChains']:
            c_chain = line.strip()
            param_values[j, :] = np.fromstring(c_chain[1:len(c_chain)-1], sep=' ')
            j += 1

        # Compute sample covariance (to be written in output.dat)
        self.cov_c = np.cov(param_values, rowvar=False)
  

    def compute_multivariate_normal(self):

        mv_norm = np.zeros(self.n_param)
        for j in range(self.n_param):
            mv_norm[j] = random.gauss(0, 1)

        # Using scipy multivariate is slower 
        # mv_norm = scipy.stats.multivariate_normal.rvs(np.zeros(self.n_param), np.eye(self.n_param))

        return mv_norm



    def terminate_loop(self):

        print("\nEnd time {}" .format(time.asctime(time.localtime())))
        print("Elapsed time: {} sec".format(time.strftime(
            "%H:%M:%S", time.gmtime(time.clock()-self.t1))))
        if self.nIterations == 0: 
            rejection_rate = 0
        else: 
            rejection_rate = self.n_rejected/self.nIterations*100
            print("Rejection rate is {} %".format(rejection_rate))
      
        self.IO_fileID['out_data'].write("$RandomVarName$\n")
        for i in range(self.n_param): 
            #self.IO_fileID['out_data'].write("X{} ".format(i))
            self.IO_fileID['out_data'].write(self.prob_distr.name_random_var[i]+' ') 

        self.IO_fileID['out_data'].write("\n$IterationNumber$\n{}\n".format(self.nIterations))
        
        self.IO_fileID['out_data'].write("\nRejection rate is {} % \n".format(rejection_rate))
        self.IO_fileID['out_data'].write("Maximum Likelihood Estimator (MLE) \n")
        #self.IO_fileID['out_data'].write("{} \n".format(self.arg_max_LL))
        self.IO_fileID['out_data'].write("Log-likelihood value \n")
        #self.IO_fileID['out_data'].write("{}".format(self.max_LL))
        self.IO_fileID['out_data'].write("\nCovariance Matrix is \n{}".format(self.cov_c))

    def compute_time(self, t1):
        """ Return the time in H:M:S from time t1 to current clock time """

        print("Estimated time: {}; accptance ratio: {}".format(time.strftime("%H:%M:%S",
                                                time.gmtime((time.clock()-t1) / float(self.it) * self.nIterations)), self.mean_r/self.it), end='\r', flush=True)


    def write_fun_distr_val(self, current_it):
        if current_it % (self.save_freq) == 0:
            #self.prob_distr.save_value(self.distr_output_file_name+"{}".format(current_it))
            self.prob_distr.save_value(current_it)

class AdaptiveMetropolisHastings(MetropolisHastings):

    def __init__(self, IO_fileID, caseName, nIterations, param_init, V, prob_distr, 
            starting_it, updating_it, eps_v):
        MetropolisHastings.__init__(self, IO_fileID, caseName, nIterations, param_init, V, prob_distr)
        self.starting_it = starting_it
        self.updating_it = updating_it
        self.S_d = 2.38**2/self.n_param
        self.eps_Id = eps_v*np.eye(self.n_param)
        self.V_i = self.V_0 
        self.X_av_i = self.param_init 

    def run_algorithm(self):

        for self.it in range(1, self.nIterations+1):

            self.compute_new_val()
            self.compute_acceptance_ratio()
            self.accept_reject()
            self.adapt_covariance(self.it)
            # self.update_sigma()
            
            # We estimate time and monitor acceptance ratio 
            self.mean_r = self.mean_r + self.r 
            if self.it % (self.nIterations/100) == 0:
                self.compute_time(self.t1)

            # Write the sample values and function evaluation in a file 
            self.prob_distr.save_sample(self.IO_fileID, self.current_val)
            self.write_fun_distr_val(self.it)

        self.compute_covariance()

        self.terminate_loop()

    def adapt_covariance(self, i):
        
        # if i >= self.starting_it:
        #     # Initialisation
        #     if i == self.starting_it:
        #         # Load all the previous iterations
        #         param_values = np.zeros((self.starting_it, self.n_param))
        #         j = 0
        #         self.IO_fileID['MChains'].seek(0)
        #         for line in self.IO_fileID['MChains']:
        #             c_chain = line.strip()
        #             param_values[j, :] = np.fromstring(c_chain[1:len(c_chain)-1], sep=' ')
        #             j += 1

        #         # Compute current sample mean and covariance
        #         self.X_av_i = np.mean(param_values, axis=0)
        #         self.V_i = self.S_d*(np.cov(param_values, rowvar=False) + self.eps_Id)

        X_i = self.current_val

        # Recursion formula to compute the mean based on previous value
        # X_av_ip = (1/(i+2))*((i+1)*self.X_av_i + X_i)
        # Formula from smith 
        X_av_ip = X_i + (i)/(i+1) * (self.X_av_i - X_i) 
        # # Mine 
        # X_av_ip = 1/(i+1) * (X_i + self.X_av_i * i)

        # Recursion formula to compute the covariance V (Haario, Saksman, Tamminen, 2001)
        #V_ip = (i/(i+1))*self.V_i + (self.S_d/(i+1))*(self.eps_Id + (i+1)*(np.tensordot(np.transpose(self.X_av_i), self.X_av_i, axes=0)) -
              #                              (i+2)*(np.tensordot(np.transpose(X_av_ip), X_av_ip, axes=0)) + np.tensordot(np.transpose(X_i), X_i, axes=0))
        # Formula from smith 
        V_ip = (i - 1)/i * self.V_i + self.S_d/i * (i*np.tensordot(np.transpose(self.X_av_i), self.X_av_i, axes=0)- (i + 1) * np.tensordot(np.transpose(X_av_ip), X_av_ip, axes=0)+ np.tensordot(np.transpose(X_i), X_i, axes=0) + self.eps_Id)

        # Update mean and covariance
        self.V_i = V_ip
        self.X_av_i = X_av_ip

        # The new value for the covariance is updated only every updating_it iterations
        if i % (self.updating_it) == 0:
        #Load all the previous iterations
            # param_values = np.zeros((i, self.n_param))
            # j = 0
            # self.IO_fileID['MChains'].seek(0)
            # for line in self.IO_fileID['MChains']:
            #     c_chain = line.strip()
            #     param_values[j, :] = np.fromstring(c_chain[1:len(c_chain)-1], sep=' ')
            #     j += 1

            # Compute current sample mean and covariance
            # self.X_av_i = np.mean(param_values, axis=0)
            # self.V_i = self.S_d*(np.cov(param_values, rowvar=False) + self.eps_Id)

            self.update_covariance()

    def update_covariance(self):
        self.R = linalg.cholesky(self.V_i)


class DelayedRejectionMetropolisHastings(MetropolisHastings):

    def __init__(self, IO_fileID, caseName, nIterations, param_init, V, prob_distr, gamma):
        MetropolisHastings.__init__(self, IO_fileID, caseName, nIterations, param_init, V, prob_distr)
        self.gamma = gamma

        # Compute inverse of covariance
        inv_R = linalg.inv(self.R)
        self.inv_V = inv_R*np.transpose(inv_R)

    def accept_reject(self):

        # Update
        u = random.random()  # Uniformly distributed number in the interval [0,1)
        if u < self.alpha:  # Accepted
            self.current_val[:] = self.new_val[:]
            self.distr_fun_current_val = self.distr_fun_new_val
            self.prob_distr.update_eval()
        else:  # Delayed rejection
            self.delayed_rejection()

    def delayed_rejection(self):

        # Delayed rejection algorithm

        # New Guess parameter (candidate or proposal)
        self.z_k = self.compute_multivariate_normal()

        # Compute new value 
        DR_new_val = self.current_val[:] + np.transpose(self.gamma*np.matmul(self.R, np.transpose(self.z_k)))

        # Acceptance ratio
        DR_distr_fun_new_eval = self.distr_fun(DR_new_val)
        r_12 = np.exp(self.distr_fun_new_val - DR_distr_fun_new_eval)
        alpha_12 = min(1, r_12)
        diff1 = self.new_val - DR_new_val 
        diff2 = self.new_val - self.current_val 
        M1 = np.matmul(diff1, self.inv_V)
        M2 = np.matmul(M1, np.transpose(diff1))
        M3 = np.matmul(diff2, self.inv_V)
        M4 = np.matmul(M3, np.transpose(diff2))
        r_2 = np.exp(DR_distr_fun_new_eval - self.distr_fun_current_val) * \
                        (np.exp(-1/2*M2) / np.exp(-1/2*M4)) * (1 - alpha_12) / (1 - self.alpha)
        #print(alpha_12, self.alpha )
        alpha_2 = min(1, r_2)

        # Update 2
        u = random.random()  # uniformly distributed number in the interval [0,1]
        if u < alpha_2:  # Accepted
            self.current_val[:] = DR_new_val[:]
            self.distr_fun_current_val = DR_distr_fun_new_eval
            self.prob_distr.update_eval()
        else:  # Rejected, current val remains the same
            self.n_rejected += 1


class DelayedRejectionAdaptiveMetropolisHastings(AdaptiveMetropolisHastings, DelayedRejectionMetropolisHastings):

    def __init__(self, IO_fileID, caseName, nIterations, param_init, V, prob_distr,
            starting_it, updating_it, eps_v, gamma):
        # There is still a problem here as we initialize two times the mother class MetropolisHastings, while once is enough. Don't know yet how to do.
        AdaptiveMetropolisHastings.__init__(self, IO_fileID, caseName, nIterations, param_init, V, prob_distr,
                                            starting_it, updating_it, eps_v)
        DelayedRejectionMetropolisHastings.__init__(self, IO_fileID, caseName, nIterations, param_init, V, prob_distr, 
                                                    gamma)

    def update_covariance(self):
        self.R = linalg.cholesky(self.V_i)
        # The inverse of the covariance must be update accordingly for the DRAM
        inv_R = linalg.inv(self.R)
        self.inv_V = np.transpose(inv_R)*inv_R


class ito_SDE(MetropolisHastings):
    """Implementation of the Ito-SDE (Arnst et al.) mcmc methods. Joffrey Coheur 17-04-19."""

    def __init__(self, IO_fileID, caseName, nIterations, param_init, prob_distr, h, f0):
        MetropolisHastings.__init__(self, IO_fileID, caseName, nIterations, param_init, np.ones([1,1]), prob_distr)

        # Time step h and free parameter f0 for the ito-sde resolution
        self.h = h
        self.f0 = f0 
        self.hfm = 1 - self.h*self.f0/4
        self.hfp = 1 + self.h*self.f0/4

        # Definition of the log-distribution function
        self.log_like = self.distr_fun

        #self.log_prior = lambda cv_xi : np.log(np.exp(cv_xi[0]) * np.exp(cv_xi[1]) * 1/((sec(1/2 + np.arctan(cv_xi[2])/np.pi))**2))

        # Compute matrix G for precondictioning (scaling and correlation)
        G_matrix = "Hessian" # Insert this in input file 
        if G_matrix == "Hessian": 
            hess_FD = -computeHessianFD(self.log_like, self.param_init, eps=0.0000001)
            self.C_approx = linalg.inv(hess_FD)  # np.array([np.log(1e4)**2])
        elif G_matrix == "Identity": 
            self.C_approx = np.identity(self.n_param)
        else: 
            raise ValueError('Unknown matrix type "{}" for estimating the conditioning matrix G .'.format(G_matrix))
 
        L_c = linalg.cholesky(self.C_approx)
        self.inv_L_c_T = linalg.inv(np.transpose(L_c))

        # Initial parameters
        self.xi_nm = np.array(self.param_init)
        self.P_nm = np.zeros(self.n_param)

    def run_algorithm(self):

        for self.it in range(1, self.nIterations+1):

            # Solve Ito-SDE using Stormer-Verlet scheme
            WP_np = np.sqrt(self.h) * self.compute_multivariate_normal()
            xi_n = self.xi_nm + self.h/2*np.matmul(self.C_approx, self.P_nm)

            # Gradient Log-Likelihood
            grad_LL = computeGradientFD(self.log_like, xi_n, eps=0.000001)
            grad_phi = - grad_LL

            P_np = self.hfm/self.hfp*self.P_nm + self.h/self.hfp * \
                (-grad_phi) + np.sqrt(self.f0)/self.hfp * \
                            np.matmul(self.inv_L_c_T, WP_np)
            xi_np = xi_n + self.h/2*np.matmul(self.C_approx, P_np)

            self.P_nm = P_np
            self.xi_nm = xi_np

            # We estimate time and monitor acceptance ratio 
            self.mean_r = self.it
            if self.it % (self.nIterations/100) == 0:
                self.compute_time(self.t1)

            # Update values 
            self.z_k=P_np 
            self.prob_distr.update_eval()

            # Write the sample values and function evaluation in a file 
            self.prob_distr.save_sample(self.IO_fileID, xi_n)
            self.write_fun_distr_val(self.it)
            # Write the standard gaussian normal proposal value, whatever is was accepted or rejected 
            # self.IO_fileID['gp'].write("{}\n".format(str(self.z_k).replace('\n', '')))

        self.compute_covariance()

        self.terminate_loop()


def computeGradientFD(f_X, var_model, schemeType='Forward', eps=0.01):
    """ ComputeGradientFD
    Compute the gradient of the model f_X with respect to its variables
    around the values provided in var_model.

    f_X is a function with variables var_model
    schemeType is optional. Default value is 'Forward' FD scheme
    eps is optional. Default value is 1/100 of the variable. 

    Joffrey Coheur 19-04-19"""

    nVar = var_model.size
    grad_FD = np.zeros(nVar)

    if schemeType == 'Forward':
        f_X_init = f_X(var_model)
        for i in range(nVar):
            # initialize parameter
            var_model_pert = np.array(var_model)
            delta_p = var_model_pert[i]*eps

            # Perturb the good parameters
            var_model_pert[i] = var_model[i]+delta_p

            # Compute the function with perturbed parameters
            f_X_pert = f_X(var_model_pert)

            # Compute the gradient using Forward FD
            grad_FD[i] = (f_X_pert - f_X_init)/delta_p

    elif schemeType == 'Central':
        for i in range(nVar):
            # initialize parameter
            var_model_pert_p = var_model
            var_model_pert_m = var_model
            delta_p = var_model_pert[i]*eps

            # Perturb the good parameters
            var_model_pert_p[i] = var_model[i]+delta_p
            var_model_pert_m[i] = var_model[i]-delta_p

            # Compute the function with perturbed parameters
            f_X_pert_p = f_X(var_model_pert_p)
            f_X_pert_m = f_X(var_model_pert_m)

            # Compute the gradient using Central FD
            grad_FD[i] = (f_X_pert_p - f_X_pert_m)/(2*delta_p)

    else:
        raise ValueError(
            "Finite difference scheme {} not implemented. \n ".format(schemeType))

    return grad_FD


def computeHessianFD(f_X, var_model, eps=0.01):
    """Compute the hessian matrix of the model f_X with respect to its variables around the values provided in var_model

    f_x is a function handle of variables var_model
    delta_p is optional. Default value of sqrt(eps) is used

    Joffrey Coheur 19-04-19"""

    nVar = var_model.size
    hess_FD = np.zeros([nVar, nVar])

    # 1) Diagonal terms
    f_X_init = f_X(var_model)
    for i in range(nVar):

        # Initialize parameter
        var_model_pert_p = np.array(var_model)
        var_model_pert_m = np.array(var_model)
        delta_pi = var_model[i]*eps

        # Perturb the good parameters
        var_model_pert_p[i] = var_model[i]+delta_pi
        var_model_pert_m[i] = var_model[i]-delta_pi

        # Compute the function with perturbed parameters
        f_X_pert_p = f_X(var_model_pert_p)
        f_X_pert_m = f_X(var_model_pert_m)

        # Finite difference scheme for the diagonal terms
        num = f_X_pert_p - 2*f_X_init + f_X_pert_m
        hess_FD[i, i] = num/(delta_pi**2)

    # 2) Off diagonal terms
    for i in range(nVar):
        # Initialize parameter
        var_i_model_pert_p = np.array(var_model)
        var_i_model_pert_m = np.array(var_model)
        delta_pi = var_model[i]*eps

        # Perturb the good parameters
        var_i_model_pert_p[i] = var_model[i]+delta_pi
        var_i_model_pert_m[i] = var_model[i]-delta_pi

        # Compute the function with perturbed parameters
        f_X_i_pert_p = f_X(var_i_model_pert_p)
        f_X_i_pert_m = f_X(var_i_model_pert_m)

        for j in range(nVar):

            if j <= i:
                continue

            delta_pj = var_model[j]*eps
            var_j_model_pert_p = np.array(var_model)
            var_j_model_pert_m = np.array(var_model)
            var_j_model_pert_p[j] = var_model[j]+delta_pj
            var_j_model_pert_m[j] = var_model[j]-delta_pj

            var_ij_model_pert_p = var_i_model_pert_p
            var_ij_model_pert_m = var_i_model_pert_m
            var_ij_model_pert_p[j] = var_i_model_pert_p[j]+delta_pj
            var_ij_model_pert_m[j] = var_i_model_pert_m[j]-delta_pj

            # Compute the function with perturbed parameters
            f_X_j_pert_p = f_X(var_j_model_pert_p)
            f_X_j_pert_m = f_X(var_j_model_pert_m)

            # Compute the function with perturbed parameters
            f_X_ij_pert_p = f_X(var_ij_model_pert_p)
            f_X_ij_pert_m = f_X(var_ij_model_pert_m)

            # Finite difference scheme for off-diagonal terms
            hess_FD[i, j] = (f_X_ij_pert_p - f_X_i_pert_p - f_X_j_pert_p + 2*f_X_init -
                            f_X_i_pert_m - f_X_j_pert_m + f_X_ij_pert_m)/(2*delta_pj*delta_pi)
            hess_FD[j, i] = hess_FD[i, j]

    return hess_FD
