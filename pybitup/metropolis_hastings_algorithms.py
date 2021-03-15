import numpy as np
import random
import scipy
from scipy import linalg
from scipy.linalg import polar
from scipy.linalg import eigh
import time
from functools import wraps
import pandas as pd 

def time_it(func):
    # decorator to time the execution time of a function. simply use @time_it before the function
    @wraps(func)
    def _time_it(*args, **kwargs):
        print("Start time {}" .format(time.asctime(time.localtime())))
        t1 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print("End time {}" .format(time.asctime(time.localtime())))
            print("Elapsed time: {} sec".format(time.strftime(
            "%H:%M:%S", time.gmtime(time.perf_counter()-t1))))

    return _time_it

class MetropolisHastings:
    """
    A generic class used to sample using Metroplis-Hastings algorithms. 
    The idea is to have the basic method that is the random-walk Markov Chain method on which is built the others. 
    

    Attributes
    ----------
    IO_fileID : dict 
        Contains the file objects to write output data. The keys are defined in solved problems. 
    caseName : char 
        Name of the case study. 
    nIterations : int
        Number of iterations in the MCMC loop.
    param_init : numpy array
        Initial values of the parameters.
    V: numpy array
        Covariance matrix used in the proposal function.
    prob_distr: BayesianPosterior 
        Contains the Bayesian posterior. It is a ProbabilityDistribution object. 


    Methods
    -------------
    run_algorithm(self)
        Contains the Metropolis-Hastings loop.
    compute_new_val(self)
        Computes the new value of the sample.
    compute_acceptance_ratio(self)
        Computes the acceptance ratio.
    accept_reject(self)
        Samples from a uniform distribution and decides whether the new sample is accepted or rejected. 
    update_sigma(self)
        Estimates the standard deviation and update it for the next step. In development. 
    update_sigma_Gelman(self)
        See "update_sigma". Uses the formula from Gelman. 
    compute_covariance(self, n_iterations)
        Computes the finale covariance. 
    compute_multivariate_normal(self)
        Samples from a multivariate normal distribution. Is used in compute_new_val 
    terminate_loop(self)
        Terminates the MCMC loop. It prints info about the result of the sampling and writes the outputs. 
    compute_time(self, t1)
        Computes the time elapsed since t1. Is used to estimate the computation time of the MCMC loop. 
    write_fun_distr_val(self, current_it)
        Write the value of the function evaluation at current_it. 
    """

    def __init__(self, IO_util, nIterations, param_init, V, prob_distr, opt_arg):
        self.nIterations = nIterations
        self.save_eval_freq = opt_arg["save_eval_freq"] 
        self.estimate_max_distr = opt_arg["estimate_max_distr"] 
        self.bool_estimate_sigma = opt_arg["estimate_sigma"] # TODO: it is not yet estimated. Value of sigma is still fixed. 

        self.V = V
        self.prob_distr = prob_distr 
        self.distr_fun = prob_distr.compute_log_value
        self.grad_log_like = prob_distr.compute_grad_log_value

        # ------------
        # Optimization 
        # ------------
        #
        # Start with an optimization problem to improve initial guess. So far, gradients must be available and only Gradient-Descent method
        # Future developments : use numerical computation of gradients, more efficient algorithm (genetic), interface with optimizer in python (SPOTPY?)
        optim = False
        if optim == "Gradient-Descent": 
            self.param_init = self.prob_distr.estimate_init_val(param_init)  
        else: 
            self.param_init = param_init

        # Initialize parameters and functions
        self.n_param = self.param_init.size
        self.current_val = self.param_init
        self.new_val = np.zeros(self.n_param)
        self.z_k = np.zeros(self.n_param)
        self.distr_fun_current_val = self.distr_fun(self.current_val)
        self.distr_grad_log_like_current_val = 1 
        self.distr_grad_log_like_new_val = 1


        # -------------------------
        # Initial variance estimate 
        # ------------------------- 
        # 
        # See Smith. 
        # prob_distr.likelihood.sum_of_square(self.current_val)
        # for model_id in prob_distr.likelihood.data.keys():
        #     sigma_k_square = prob_distr.likelihood.SS_X / (prob_distr.likelihood.data[model_id].num_points - self.n_param)
        #     prob_distr.likelihood.data[model_id].std_y = np.sqrt(sigma_k_square)


        # Outputs 
        self.IO_util = IO_util
        self.IO_fileID = IO_util["fileID"]
        # Ensure that we start the files at 0 (due to the double initialisation of DRAM)
        self.IO_fileID['MChains'].seek(0) 
        self.IO_fileID['MChains_reparam'].seek(0)

        # Write initial values 
        self.prob_distr.save_sample(self.IO_fileID, self.current_val)
        self.prob_distr.update_eval()
        self.write_fun_distr_val(0)

        # Write in output.dat parameters info and iteration number 
        self.IO_fileID['out_data'].write("$RandomVarName$\n")
        for i in range(self.n_param): 
            #self.IO_fileID['out_data'].write("X{} ".format(i))
            self.IO_fileID['out_data'].write(self.prob_distr.name_random_var[i]+' ') 
        self.IO_fileID['out_data'].write("\n$IterationNumber$\n{}\n".format(self.nIterations))

        # Variables for estimating the MAP
        self.arg_MAP = np.array(self.current_val[:]) 
        self.MAP_val = self.distr_fun_current_val

        # Cholesky decomposition of V. Constant if MCMC is not adaptive
        self.V_0 = V 
        self.R = linalg.cholesky(V)

        # Monitoring the chain
        self.n_rejected = 0

        # Start the clock count for estimating the time for the total numer of iterations
        self.t1 = time.perf_counter()

        self.it = 0 

        # Monitoring acceptation rate 
        self.mean_r = 0
        self.r = 0

        
    @time_it
    def run_algorithm(self):

        for self.it in range(1, self.nIterations+1):
            self.compute_new_val()
            self.compute_acceptance_ratio()
            self.accept_reject()
            #self.update_sigma() 
            #self.update_sigma_constant()
            #self.update_sigma_Gelman()

            # We estimate time and monitor acceptance ratio 
            self.mean_r = self.mean_r + self.alpha 
            if self.it % (self.nIterations/1000) == 0:
                self.compute_time(self.t1)

            # Write the sample values and function evaluation in a file 
            #self.prob_distr.save_log_post(self.IO_fileID)
            self.prob_distr.save_sample(self.IO_fileID, self.current_val)
            self.write_fun_distr_val(self.it)


        self.compute_covariance(self.nIterations+2)

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
        # print(self.distr_fun_new_val, self.distr_fun_current_val)
        # print(self.distr_fun_new_val - self.distr_fun_current_val)
        if self.distr_fun_new_val == -np.inf: 
            self.r = 0
        else: 
            #self.r = self.distr_fun_new_val/self.distr_fun_current_val
            
            self.r = np.exp(self.distr_fun_new_val - self.distr_fun_current_val)

        self.alpha = min(1, self.r)
        #print(self.r, self.alpha)
        
    def accept_reject(self):

        # Update
        u = random.random()  # Uniformly distributed number in the interval [0,1)
        if u < self.alpha:  # Accepted
            self.current_val[:] = self.new_val[:]

            # Estimate the MAP here 
            self.estimate_max()

            self.distr_fun_current_val = self.distr_fun_new_val
            self.distr_grad_log_like_current_val = self.distr_grad_log_like_new_val
            self.prob_distr.update_eval()
        else:  # Rejected, current val remains the same
            self.n_rejected += 1

    def estimate_max(self): 
        """ Estimate the maximum of the distribution and the arg max.
        Useful in Bayesian inference context, where this will estimate the MAP 
        in line with the MCMC iterations. """

        if self.estimate_max_distr is True:
            if self.distr_fun_new_val > self.MAP_val:
                self.arg_MAP[:] = self.new_val[:]
                self.MAP_val = self.distr_fun_new_val
                print("\nNew max. distr. value {} (in log) found at iteration {}.".format(self.MAP_val, self.it))
        else: 
            return 

    def update_sigma(self): 
        """ Smith, (2013)."""
        ns = 0.01
        self.prob_distr.likelihood.sum_of_square(self.current_val) #update SS_X 
        for model_id in self.prob_distr.likelihood.data.keys():

            #print(self.prob_distr.likelihood.data[model_id].std_y)
            a = 0.5 * (ns + self.prob_distr.likelihood.data[model_id].n_runs) 

            for n in range(int(self.prob_distr.likelihood.data[model_id].num_points)):
                # self.prob_distr.likelihood.data[model_id].var_s[n] 
                
                b = 0.5 * (ns * self.prob_distr.likelihood.data[model_id].var_s[n]  + self.prob_distr.likelihood.SS_X[n]/self.prob_distr.likelihood.data[model_id].n_runs)
                self.prob_distr.likelihood.data[model_id].std_y[n] = np.sqrt(scipy.stats.invgamma.rvs(a, loc=0, scale=b))
                #if self.prob_distr.likelihood.data[model_id].std_y[n] < 1e-4: 
                self.prob_distr.likelihood.data[model_id].std_y[n] = self.prob_distr.likelihood.data[model_id].n_runs * self.prob_distr.likelihood.data[model_id].std_y[n]
            #print(self.prob_distr.likelihood.data[model_id].std_y)
            self.prob_distr.likelihood.data[model_id].std_s += self.prob_distr.likelihood.data[model_id].std_y / self.prob_distr.likelihood.data[model_id].n_runs

    def update_sigma_constant(self): 
        """ Smith, (2013). Sigma is a constant with time """
        ns = 0.01
        self.prob_distr.likelihood.sum_of_square(self.current_val) #update SS_X 

        for model_id in self.prob_distr.likelihood.data.keys():

            #print(self.prob_distr.likelihood.data[model_id].std_y)
            a = 0.5 * (ns + self.prob_distr.likelihood.data[model_id].num_points) 
            b = 0.5 * (ns * self.prob_distr.likelihood.data[model_id].std_y[0]**2  + np.sum(self.prob_distr.likelihood.SS_X))

            rv_sigma = np.sqrt(scipy.stats.invgamma.rvs(a, loc=0, scale=b))
            for n in range(int(self.prob_distr.likelihood.data[model_id].num_points)):
                self.prob_distr.likelihood.data[model_id].std_y[n] = rv_sigma




    def update_sigma_Gelman(self): 
        """ Gelman Bayesian Data Analysis, 3rd Edition.
        Sigma is in the family of inverse chi squared distribution, which is equivalent to a 
        inverse gamma with an other parametrization. (see wikipedia, scaled inverse chi-squared distribution 
        for the relation between the two distributions. """ 
        
        self.prob_distr.likelihood.sum_of_square(self.current_val) #update SS_X 
        for model_id in self.prob_distr.likelihood.data.keys():

            #print(self.prob_distr.likelihood.data[model_id].std_y)
            a = 0.5 * (self.prob_distr.likelihood.data[model_id].n_runs - 1) 

            for n in range(int(self.prob_distr.likelihood.data[model_id].num_points)):
                # self.prob_distr.likelihood.data[model_id].var_s[n] 
                b = a * self.prob_distr.likelihood.data[model_id].var_s[n]
                self.prob_distr.likelihood.data[model_id].std_y[n] = np.sqrt(scipy.stats.invgamma.rvs(a, loc=0, scale=b) + 1e-10)  
            #print(self.prob_distr.likelihood.data[model_id].std_y)
            self.prob_distr.likelihood.data[model_id].std_s += self.prob_distr.likelihood.data[model_id].std_y 

    def compute_covariance(self, n_iterations):

        file_path = self.IO_fileID['MChains_reparam'] # MChains, MChains_reparam

        # Load all the previous iterations
        param_values = np.zeros((n_iterations, self.n_param))
        j = 0
        file_path.seek(0) 
        for line in file_path:
            c_chain = line.strip()
            param_values[j, :] = np.fromstring(c_chain[1:len(c_chain)-1], sep=' ')
            j += 1

        # Compute sample mean 
        self.mean_c = np.mean(param_values, axis=0)

        # Compute sample covariance (to be written in output.dat)
        self.cov_c = np.cov(param_values, rowvar=False)
        np.savetxt("cov.csv", self.cov_c, delimiter=",")
  

    def compute_multivariate_normal(self):

        mv_norm = np.zeros(self.n_param)
        for j in range(self.n_param):
            mv_norm[j] = random.gauss(0, 1)

        # Using scipy multivariate is slower 
        # mv_norm = scipy.stats.multivariate_normal.rvs(np.zeros(self.n_param), np.eye(self.n_param))

        return mv_norm


    def terminate_loop(self):

        if self.nIterations == 0: 
            rejection_rate = 0
        else: 
            rejection_rate = self.n_rejected/self.nIterations*100
            print("\nRejection rate is {} %".format(rejection_rate))

        # TODO : write the estimation of sigma in a file. Only when there is 
        # a likelihood function. Put this in opt_arg 
        if self.bool_estimate_sigma is True: 
            for model_id in self.prob_distr.likelihood.data.keys():
                est_sigma = self.prob_distr.likelihood.data[model_id].std_y
                #print("\n Experimental error std is {} %".format(est_sigma))

                myfile = {'model_id': est_sigma} 
                df = pd.DataFrame(myfile, columns=['model_id'])
                df.to_csv(self.IO_fileID['estimated_sigma'])
        
        self.IO_fileID['out_data'].write("\nRejection rate is {} % \n".format(rejection_rate))

        self.IO_fileID['out_data'].write("\nElapsed time: {} \n".format(time.strftime("%H:%M:%S",time.gmtime(time.perf_counter()-self.t1))))
        
        # TODO: Estimate MLE during MCMC iterations and save it
        self.IO_fileID['out_data'].write("\nMaximum Likelihood Estimator (MLE) \n")
        #self.IO_fileID['out_data'].write("{} \n".format(self.arg_max_LL))
        self.IO_fileID['out_data'].write("Log-likelihood value \n")
        #self.IO_fileID['out_data'].write("{}".format(self.max_LL))
        
        # Write in a csv the estimation of the MAP and arg MAP 
        if self.estimate_max_distr is True: 
            
            np.savetxt(self.IO_fileID['estimate_arg_max_val_distr'], np.array([self.prob_distr.model.parametrization_backward(self.arg_MAP[:])]),fmt="%f", delimiter=",")
            np.savetxt(self.IO_fileID['estimate_max_val_distr'], np.array([np.exp(self.MAP_val)]),fmt="%f", delimiter=",")

        self.IO_fileID['out_data'].write("\nCovariance Matrix is \n{}".format(self.cov_c))

    def compute_time(self, t1):
        """ Return the time in H:M:S from time t1 to current clock time """

        print("\rIteration {}/{} ({:.2f}%); Remaining time: {}; mean acceptance probability: {}\t".format(self.it, self.nIterations, self.it/self.nIterations*100, time.strftime("%H:%M:%S",
                                                time.gmtime((time.perf_counter()-t1) / float(self.it) * self.nIterations - (time.perf_counter()-t1) )), self.mean_r/self.it), end='', flush=True)


    def write_fun_distr_val(self, current_it):
        if current_it % (self.save_eval_freq) == 0:
            self.prob_distr.save_value(self.IO_util, current_it)

class AdaptiveMetropolisHastings(MetropolisHastings):
    """
    A class used to sample using the adaptive random-walk Metroplis-Hastings algorithm (AMH). 
    It is based on the MetropolisHastings class. 
    

    Attributes
    ----------
    starting_it : int 
        Iteration at which the covariance adaptation starts 
    updating_it : int
        The frequency at which to update the covariance estimate 
    eps_v : double
        Correcting factor to ensure ergodic properties of the algorithm 


    Methods
    -------------
    run_algorithm(self)
        Contains the Metropolis-Hastings loop with the adaptation step.
    adapt_covariance(self, i)
        Computes a new estimation of the covariance matrix at the i-th iteration.  
    update_covariance(self)
        Updates the covariance value to be used at the the next iteration.   

    """

    def __init__(self, IO_fileID, nIterations, param_init, V, prob_distr, 
            starting_it, updating_it, eps_v, opt_arg):
        MetropolisHastings.__init__(self, IO_fileID, nIterations, param_init, V, prob_distr, opt_arg)
        self.starting_it = starting_it
        self.updating_it = updating_it
        self.S_d = 2.38**2/self.n_param
        self.eps_Id = eps_v*np.eye(self.n_param)
        self.V_i = self.V_0 
        self.X_av_i = self.param_init 

    @time_it
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
            #self.prob_distr.save_log_post(self.IO_fileID)
            self.prob_distr.save_sample(self.IO_fileID, self.current_val)
            self.write_fun_distr_val(self.it)

        self.compute_covariance(self.nIterations+2)

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
    """
    A class used to sample using the delayed-rejection Metroplis-Hastings algorithm (DR). 
    It is based on the MetropolisHastings class. 
    In this version of the DR algorithm, only one delayed-rejection is implemented. 
    TODO: generalized to multiple delayed rejections. 
    

    Attributes
    ----------
    gamma : int 
        Controls the size of the new jumps when the rejection of a sample has been delayed. 
    
    Methods
    -------------
    accept_reject(self):
        Samples from a uniform distribution and decides whether the new sample is accepted or rejected.
        If it is rejected, then the rejection is delayed and we go to the new delayed_rejection step. 
    delayed_rejection(self): 
        Computes the new acceptance probability and decides if the new proposed sample is accepted or rejected. 
    """

    def __init__(self, IO_fileID, nIterations, param_init, V, prob_distr, gamma, opt_arg):
        MetropolisHastings.__init__(self, IO_fileID, nIterations, param_init, V, prob_distr, opt_arg)
        self.gamma = gamma

        # Compute inverse of covariance
        inv_R = linalg.inv(self.R)
        self.inv_V = inv_R*np.transpose(inv_R)

    def accept_reject(self):

        # Update
        u = random.random()  # Uniformly distributed number in the interval [0,1)
        if u < self.alpha:  # Accepted
            self.current_val[:] = self.new_val[:]

            # Estimate the MAP here 
            self.estimate_max()

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
    """
    A class used to sample using the delayed-rejection Metroplis-Hastings algorithm (DRAM). 
    It is based on the AdaptiveMetropolisHastings and DelayedRejectionMetropolisHastings classes.  

    Attributes
    ----------    
    No new attributes. 

    Methods
    -------------
    update_covariance(self):
        Update the value of the covariance and its inverse that is used in the delayed-rejection step. 
  
    """

    def __init__(self, IO_fileID, nIterations, param_init, V, prob_distr, starting_it, updating_it, eps_v, gamma, opt_arg):
        # There is still a problem here as we initialize two times the mother class MetropolisHastings, while once is enough. Don't know yet how to do.
        AdaptiveMetropolisHastings.__init__(self, IO_fileID, nIterations, param_init, V, prob_distr, starting_it, updating_it, eps_v, opt_arg)
        DelayedRejectionMetropolisHastings.__init__(self, IO_fileID, nIterations, param_init, V, prob_distr, gamma, opt_arg)

    def update_covariance(self):
        self.R = linalg.cholesky(self.V_i)
        # The inverse of the covariance must be update accordingly for the DRAM
        inv_R = linalg.inv(self.R)
        self.inv_V = np.transpose(inv_R)*inv_R


class GradientBasedMCMC(MetropolisHastings): 
    """
    A generic class for the gradient-based sampling methods. 
    It is based on the MetropolisHastings. 

    Attributes
    ----------    
    C_matrix: dict 
        Contains the input for the covariance adaptation. 
    gradient: str
        Defines whether the gradient is computed numerically or if an analytical expression is provided. 

    Methods
    -------------
    _set_algo_param(self, *args): 
        Sets the algorithm free parameters.
    adapt_covariance(self, i, *args):
        Adapts the covariance matrix. 



    """

    def __init__(self, IO_fileID, nIterations, param_init, V, prob_distr, C_matrix, gradient, opt_arg):

        MetropolisHastings.__init__(self, IO_fileID, nIterations, param_init, V, prob_distr, opt_arg)

        # Definition of the log-distribution function 
        self.log_like = self.distr_fun 
        
        # Parameters for adaptation 
        self.X_av_i = np.array(self.param_init)
        self.S_d = 1.0 # 2.38**2/self.n_param
        self.eps_Id = 1e-10*np.eye(self.n_param)
        self.starting_it = int(C_matrix['starting_it'])
        self.update_it = int(C_matrix['update_it'])
        # End of adaptation it not mandatory. Default is equal to the number of iterations
        if (C_matrix.get("end_it") is not None): 
            self.end_it = int(C_matrix['end_it'])
        else: 
            self.end_it = self.nIterations + 1


        #---------------
        # Hessian matrix
        # --------------
        #  
        # Estimation of the normalisation matrix. Type of approx for C is either Hessian, Hessian_diag or Identity matrices 
        C_matrix_estimation = C_matrix['value']
        # Check for Hessian numerical computation : LL = x^2 + y^2 
        # LL = lambda Z: Z[0]**2 + Z[1]**2 
        # hess_FD = computeHessianFD(LL, np.array([1., 1.]), eps=0.001)
        # print(hess_FD)

        # Compute matrix C for precondictioning (scaling and correlation)
        if C_matrix_estimation == "Hessian": 
            print('Computing full Hessian for scaling and correlation.')
            self.hess_mat = -computeHessianFD(self.log_like, self.param_init, eps=0.001)
            self.C_approx = linalg.inv(self.hess_mat)  
        elif C_matrix_estimation == "Hessian_diag": 
            print('Computing diagonal of Hessian for scaling and correlation.')
            self.hess_mat = -computeHessianFD(self.log_like, self.param_init, eps=0.001, compute_all_element=False)
            self.C_approx = linalg.inv(self.hess_mat)  
        elif C_matrix_estimation == "Identity": 
            print('Using identity matrix for Hessian approxiation')
            self.hess_mat = np.identity(self.n_param)
            self.C_approx = np.identity(self.n_param)
        elif C_matrix_estimation == "PSD": 
            print('Positive semi-definite approximation of the Hessian. Didn''t prove to be correct so far.')
            # Positive semi-definitive approximation. Approximation not accurate so far. 
            self.hess_mat = self.prob_distr.estimate_hessian_model(self.param_init)
            self.C_approx = linalg.inv(self.hess_mat)
        elif C_matrix_estimation == "nearPD": 
            print('Nearest positive semi-definite matrix. ')
            self.hess_mat = -computeHessianFD(self.log_like, self.param_init, eps=0.001)
            C_approx_nonPD = linalg.inv(self.hess_mat)
            print(C_approx_nonPD)
            self.C_approx  = nearPD(C_approx_nonPD)
        elif C_matrix_estimation == "Matrix": # THIS IS NOT WORKING YET
            # TODO : all this should probably go in the inference_problem.py, 
            # probably the same way as I did with covariance for RWMH, AMH, etc.   
            print('Initial covariance matrix provided.')
            self.C_approx = np.array(C_matrix['matrix_value']) 
        elif C_matrix_estimation == "from_file": 
            print('Reading cov.csv file.')
            reader = pd.read_csv(self.IO_util['path']['cwd']+"/cov.csv", header=None)
            self.C_approx = reader.values

            # Generalized eigenval problem for Maarten
            # print('Computing full Hessian for scaling and correlation.')
            # self.hess_mat = computeHessianFD(self.log_like, self.param_init, eps=0.001, positive_diag=False)
            print('Computing diagonal of Hessian for scaling and correlation.')
            self.hess_mat = computeHessianFD(self.log_like, self.param_init, eps=0.001, compute_all_element=False, positive_diag=True)
            M = linalg.inv(self.C_approx)
            eigvals, eigvecs = eigh(M, -self.hess_mat, eigvals_only=False)
            print("Engenvalues = {}".format(eigvals))
            print("Eigenvectors = {}".format(eigvecs))

        else: 
            raise ValueError('Unknown matrix type "{}" for estimating the conditioning matrix G .'.format(C_matrix_estimation))
        
        print("Inverse hessian determinant = {}".format(np.linalg.det(self.C_approx)))

      
        try: 
            self.L_c = linalg.cholesky(self.C_approx)
            self.inv_L_c = linalg.inv(self.L_c)
            self.inv_L_c_T = linalg.inv(np.transpose(self.L_c))
        except: 
            print('Non positive defintie Hessian matrix. Set to identity matrix.')
            self.C_approx = np.identity(self.n_param)
            self.L_c = np.identity(self.n_param)
            self.inv_L_c = np.identity(self.n_param)
            self.inv_L_c_T = np.identity(self.n_param)

        # self.L_c = linalg.cholesky(self.C_approx)
        # self.inv_L_c_T = linalg.inv(np.transpose(self.L_c))

        #print(self.hess_mat)

        # --------------------
        # Gradient computation 
        # --------------------
        # 
        # Computation of the gradient for the direction in the Ito-SDE
        # Estimation of the gradient : either Analytical or Numerical 
        if gradient == "Numerical": 
            # Numerical computation of the gradient using finite differences  
            self.computeGrad = lambda xi_n: computeGradientFD(self.log_like, xi_n, eps=0.0000000001)
        elif gradient == "Analytical": 
            # Analytical formula for the computation of gradient specified 
            self.computeGrad = self.grad_log_like
        else: 
            raise ValueError('Unknown gradient type "{}" for estimating gradients.'.format(gradient))

    def _set_algo_param(self, *args): 
        pass 

    def adapt_covariance(self, i, *args):
        """ Adapt covariance and algorithm parameters if asked. 
        args contain algorithm parameters. """ 

        if self.it >= self.starting_it and self.it <= self.end_it+1:
        
            if  self.it == self.starting_it:  
                # initialise covariance computation from all previous iterations
                self.compute_covariance(self.it) # update self.mean_c and self.cov_c
                self.X_av_i = self.mean_c
                self.V_i = self.S_d*(self.cov_c + self.eps_Id)

                print("Iteration {}: starting covariance adaptation (freq. {} iterations).".format(self.it, self.update_it))
                self.IO_fileID['out_data'].write("\nIteration {}: starting covariance adaptation (freq. {} iterations).".format(self.it, self.update_it)) 

            elif self.it <= self.end_it:
                # Compute covariance using recursion formula 
                X_i = self.current_val[:]
                X_av_ip = X_i + (self.it)/(self.it + 1) * (self.X_av_i - X_i) 
                V_ip = (self.it - 1)/self.it  * self.V_i + self.S_d/self.it  * (self.it *np.tensordot(np.transpose(self.X_av_i), self.X_av_i, axes=0)- (self.it  + 1) * np.tensordot(np.transpose(X_av_ip), X_av_ip, axes=0)+ np.tensordot(np.transpose(X_i), X_i, axes=0) + self.eps_Id)
                # Update mean and covariance
                self.V_i = V_ip
                self.X_av_i = X_av_ip

            elif self.it == self.end_it+1:  
                # End of adaptation
                print("Stopping covariance adaptation (iteration {}).".format(self.it-1))
                self.IO_fileID['out_data'].write("\nIteration {}: stopping covariance adaptation.".format(self.it)) 
                self.IO_fileID['out_data'].write("\nThe covariance matrix estimation at the end of the adaptation procedure is: \n{}".format(self.C_approx))

            # Effectively update the covariance in the MCMC algorithm 
            if  self.it % self.update_it == 0: 
                # Adapt time step 
                # if self.it == self.update_it: 
                #     self._set_algo_param(*args)
                
                # print("\nNew cov:{}\n".format(self.V_i))

                self.C_approx = self.V_i
                self.L_c = linalg.cholesky(self.C_approx)
                self.inv_L_c = linalg.inv(self.L_c)
                self.inv_L_c_T = np.transpose(self.inv_L_c)

class HamiltonianMonteCarlo(GradientBasedMCMC):
    """
    A class used to sample using the Hamiltonian Monte Carlo (HMC) algorithm.  
    It is based on the GradientBasedMCMC. 

    Algorithm: From Neal, R. M. (2010) MCMC using Hamiltonian dynamics. In Handbook of Markov Chain Monte Carlo (eds
    S. Brooks, A. Gelman, G. Jones and X.-L Meng). Boca Raton: Chapman and Hall–CRC Press. 
    Implementation: Joffrey Coheur 14-01-20.
    TODO: improvements such as the no U-turn sampler. 

    Attributes
    ----------    
    step_size : double 
        Time step in the Stormer-Verlet numerical scheme
    num_steps : in 
        Controls the damping parameter

    Methods
    -------------
    _set_algo_param(self, *args): 
        Sets the free parameters of the HMC algorithm. 
    compute_new_val(self):
        Computes the new value of the sample obtained after solving Hamiltonian dynamics. 
    compute_acceptance_ratio(self):
        Computes the acceptance ratio.
    """

    def __init__(self, IO_fileID, nIterations, param_init, prob_distr, C_matrix, gradient, step_size, num_steps, opt_arg):

        GradientBasedMCMC.__init__(self, IO_fileID, nIterations, param_init, np.ones([1,1]), prob_distr, C_matrix, gradient, opt_arg)

        # HMC tuning parameters 
        self._set_algo_param(step_size, num_steps)

        # Initial parameters
        self.current_val = self.param_init 
        self.distr_fun_current_val = self.distr_fun(self.current_val)
        self.distr_grad_log_like_current_val = self.computeGrad(self.current_val)

    def _set_algo_param(self, *args): 
        self.epsilon = args[0] 
        self.L =  args[1] 

    def compute_new_val(self):
        """ Solve Hamiltonian dynamics using leapfrog discretization. """

        # Adapt covariance 
        self.adapt_covariance(self.it, 0.05, self.L)

        #Initialise 
        self.new_val[:] = self.current_val[:]
        self.p = self.compute_multivariate_normal()
        self.p = np.matmul(self.p, self.inv_L_c_T)
        self.current_p = self.p

        np.savetxt(self.IO_fileID['aux_variables'], np.transpose(np.array([self.p, self.new_val[:]])), fmt="%f", delimiter=",")

        # Make a half step for momentum at the beginning
        self.p = self.p - self.epsilon * (-self.distr_grad_log_like_current_val) / 2

        np.savetxt(self.IO_fileID['aux_variables'], np.transpose(np.array([self.p, self.new_val[:]])), fmt="%f", delimiter=",")
        #self.IO_fileID['aux_variables'].write("{}\n".format(str([self.p, self.new_val[:]]).replace('\n', '')))

        # Alternate full steps for position and momentum 
        for i in range(self.L):
            # Make a full step for the position 
            self.new_val = self.new_val + self.epsilon * (np.matmul(self.p, self.C_approx))
            self.distr_grad_log_like_new_val = self.computeGrad(self.new_val)
            #print(computeGradientFD(self.log_like, self.new_val, eps=0.00000001), self.distr_grad_log_like_new_val) # For comparing. "gradient" must be set to Analytical 
            # Make a full step for the momentum, except at end of trajectory 
            if i!=(self.L-1): 
                self.p = self.p - self.epsilon * (-self.distr_grad_log_like_new_val)
                np.savetxt(self.IO_fileID['aux_variables'], np.transpose(np.array([self.p, self.new_val[:]])), fmt="%f", delimiter=",")
        # Make a half step for momentum at the end.
        self.p = self.p - self.epsilon * (-self.distr_grad_log_like_new_val) / 2
        np.savetxt(self.IO_fileID['aux_variables'], np.transpose(np.array([self.p, self.new_val[:]])), fmt="%f", delimiter=",")
        # Negate momentum at end of trajectory to make the proposal symmetric
        self.p = -self.p

        # Save in a "aux_variables" file the momentum variable
        #np.savetxt(self.IO_fileID['aux_variables'], np.transpose(np.array([self.p, self.new_val[:]])), fmt="%f", delimiter=",")

    def compute_acceptance_ratio(self):

        # Evaluate potential and kinetic energies at start and end of trajectory
        current_U = -self.distr_fun_current_val 
        L_current_p = np.matmul(self.current_p, self.C_approx)
        arg_current_K = L_current_p * self.current_p
        
        current_K = np.sum(arg_current_K) / 2 
        #current_K = np.sum(self.current_p**2) / 2

        self.distr_fun_new_val = self.distr_fun(self.new_val)
        proposed_U = -self.distr_fun_new_val  
        L_p = np.matmul(self.p, self.C_approx)
        arg_proposed_K = L_p * self.p
        proposed_K = np.sum(arg_proposed_K) / 2 
        #proposed_K = np.sum(self.p**2) / 2
       
        if self.distr_fun_new_val is -np.inf: 
            self.r = 0
        else: 
            #self.r = self.distr_fun_new_val/self.distr_fun_current_val
            self.r = np.exp(current_U-proposed_U+current_K-proposed_K)

        self.alpha = min(1, self.r)

class ito_SDE(GradientBasedMCMC):
    """
    A class used to sample using the Ito stochastic differential equation (ISDE) Markov Chain Monte Carlo algorithm. 
    It is based on the GradientBasedMCMC. Note that ISDE is not a Metropolis-Hastings type algorithm, but the 
    implementation of the class GradientBasedMCMC is based itself on the class MetropolisHastings.  

    Attributes
    ----------    
    h : double 
        Time step in the Stormer-Verlet numerical scheme
    f0 : double 
        Controls the damping parameter

    Methods
    -------------
    _set_algo_param(self, *args): 
        Sets the free parameters of the ISDE algorithm. 
    run_algorithm(self):
        This function is totally re-written because there is not accept-reject step in ISDE. 
  
   
    Algorithm: From Soize, Arnst et al.
    Implementation: Joffrey Coheur 17-04-19.
    """

    def __init__(self, IO_fileID, nIterations, param_init, prob_distr, C_matrix, gradient, h, f0, opt_arg):

        GradientBasedMCMC.__init__(self, IO_fileID, nIterations, param_init, np.ones([1,1]), prob_distr, C_matrix, gradient, opt_arg)

        # Time step h and free parameter f0 for the ito-sde resolution
        self._set_algo_param(h, f0)

        # Initial parameters
        self.xi_nm = np.array(self.param_init)
        self.P_nm = np.zeros(self.n_param)
        np.savetxt(self.IO_fileID['aux_variables'], np.transpose(np.array([self.P_nm, self.xi_nm])), fmt="%f", delimiter=",")
        #self.IO_fileID['aux_variables'].write("{}\n".format(str(self.P_nm).replace('\n', '')))

    def _set_algo_param(self, *args): 
        """ Set the time step h, damping parameters f0 and dependent parameters."""
        self.h = args[0]
        self.f0 = args[1]
        self.hfm = 1 - self.h*self.f0/4
        self.hfp = 1 + self.h*self.f0/4


    @time_it
    def run_algorithm(self):

        for self.it in range(1, self.nIterations+1):

            # if  self.it % 10 == 0: 
            #     print("Recomputing Hessian matrix... ")
            #     # Update Hessian 
            #     self.hess_mat = -computeHessianFD(self.log_like, self.xi_nm, eps=0.001)

            #     try: 
            #         self.C_approx = linalg.inv(self.hess_mat)  # np.array([np.log(1e4)**2])
            #         self.L_c = linalg.cholesky(self.C_approx)
            #         self.inv_L_c_T = linalg.inv(np.transpose(self.L_c))
            #     except: 
            #         print('Non positive defintie local Hessian matrix. Set to identity matrix.\n')
            #         diag_hess = np.diag(np.abs(np.diag(self.hess_mat)))
            #         print(diag_hess)
            #         self.C_approx = linalg.inv(diag_hess)
            #         self.L_c = linalg.cholesky(self.C_approx)
            #         self.inv_L_c_T = linalg.inv(np.transpose(self.L_c))
            #         # self.C_approx = np.identity(self.n_param)
            #         # self.L_c = np.identity(self.n_param)
            #         # self.inv_L_c_T = np.identity(self.n_param)

            # Adapt covariance 
            self.current_val = self.xi_nm
            self.adapt_covariance(self.it, self.h, self.f0)

            # # Update covariance with recursion 
            # if  self.it == self.update_it:  
            #     # Initialise sample mean and covariance estimation 
            #     self.compute_covariance(self.it) # update self.mean_c and self.cov_c
            #     self.X_av_i = self.mean_c
            #     self.V_i = self.S_d*(self.cov_c + self.eps_Id)

            # if self.it > self.update_it and self.it < 1e5: 
            #     X_i = self.current_val
            #     X_av_ip = X_i + (self.it)/(self.it + 1) * (self.X_av_i - X_i) 
            #     V_ip = (self.it - 1)/self.it  * self.V_i + self.S_d/self.it  * (self.it *np.tensordot(np.transpose(self.X_av_i), self.X_av_i, axes=0)- (self.it  + 1) * np.tensordot(np.transpose(X_av_ip), X_av_ip, axes=0)+ np.tensordot(np.transpose(X_i), X_i, axes=0) + self.eps_Id)
            #     # Update mean and covariance
            #     self.V_i = V_ip
            #     self.X_av_i = X_av_ip

            # # Update covariance 
            # if  self.it % self.update_it == 0 and self.it < 1e5: 
            #     # Adapt time step 
            #     if self.it == self.update_it: 
            #         # Adat time step (f0 is kept the same)
            #         self._set_algo_param(0.1, self.f0)
                
            #     print("\n{}\n".format(self.V_i))

            #     self.C_approx = self.V_i
            #     self.L_c = linalg.cholesky(self.C_approx)
            #     self.inv_L_c_T = linalg.inv(np.transpose(self.L_c))

            # Solve Ito-SDE using Stormer-Verlet scheme
            WP_np = np.sqrt(self.h) * self.compute_multivariate_normal()
            xi_n = self.xi_nm + self.h/2*np.matmul(self.C_approx, self.P_nm)

            # Gradient Log-Likelihood
            grad_phi = -self.computeGrad(xi_n)
            #print(computeGradientFD(self.log_like, xi_n, eps=0.00000001), grad_LL) # For comparing, "gradient" must be set to Analytical 
            # grad_phi = -grad_LL

            P_np = self.hfm/self.hfp*self.P_nm + self.h/self.hfp * \
                (-grad_phi) + np.sqrt(self.f0)/self.hfp * \
                            np.matmul(self.inv_L_c,WP_np) 
            xi_np = xi_n + self.h/2*np.matmul(self.C_approx, P_np)

            self.P_nm = P_np
            self.xi_nm = xi_np
            # We estimate time 
            self.mean_r = self.it
            if self.it % (self.nIterations/100) == 0:
                self.compute_time(self.t1)

            # Estimate the MAP  
            if self.estimate_max_distr is True: 
                # TODO : can improve computational cost of estimating 
                # the MAP by only running one time the model (running 
                # self.distr_fun(xi_np) call the model, while it was already computed at self.computeGrad(xi_n))
                # TODO: use the same function self.estimate_max()? 
                self.distr_fun_new_val = self.distr_fun(xi_np)
                if self.distr_fun_new_val  > self.MAP_val:
                    self.arg_MAP[:] = xi_np[:]
                    self.MAP_val = self.distr_fun_new_val
                    np.savetxt("output/arg_MAP_estimation.csv", np.array([self.prob_distr.model.parametrization_backward(self.arg_MAP[:])]),fmt="%f", delimiter=",")

                    # print(self.arg_MAP[:])
                    print("It: {}, log MAP value: {}".format(self.it, self.MAP_val))

            # Update values 
            self.z_k=P_np 
            self.prob_distr.update_eval()

            # Write the sample values and function evaluation in a file 
            #self.prob_distr.save_log_post(self.IO_fileID)
            self.prob_distr.save_sample(self.IO_fileID, xi_np)
            np.savetxt(self.IO_fileID['aux_variables'], np.transpose(np.array([self.P_nm, self.xi_nm])), fmt="%f", delimiter=",")
            #self.IO_fileID['aux_variables'].write("{}\n".format(str(self.P_nm).replace('\n', '')))
            self.write_fun_distr_val(self.it)
            # Write the standard gaussian normal proposal value, whatever is was accepted or rejected 
            # self.IO_fileID['gp'].write("{}\n".format(str(self.z_k).replace('\n', '')))

        self.compute_covariance(self.nIterations+2)

        self.terminate_loop()


def computeGradientFD(f_X, var_model, schemeType='Forward', eps=0.01):
    """ 
    A function that computes the gradient of a model with respect to its variables around a given value.

    Parameters
    ----------
    f_X: 
        Function handle 
    var_model: numpy array 
        Variables of the function f_X 
    schemeType: (optional)
        Default is 'Forward' FD scheme
    eps: double (optional)
        Default value is 1/100 of the variable. 

    Returns
    ---------
    grad_FD: numpy array
        Gradient of the function f_X with respect to the different variables at var_model

    Joffrey Coheur 19-04-19
    """

    nVar = var_model.size
    grad_FD = np.zeros(nVar)

    if schemeType == 'Forward':
        f_X_init = f_X(var_model)
        for i in range(nVar):
            # initialize parameter
            var_model_pert = np.array(var_model)
            delta_p = var_model_pert[i]*eps

            if delta_p == 0.:
                # Happens if, for instance, var_model = 0.0 
                # This should be check also for Central and Hessian computation, but not done yet 
                delta_p = eps

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


def computeHessianFD(f_X, var_model, eps=0.01, compute_all_element=True, positive_diag=True):
    """ 
    A function that computes the hessian of a model with respect to its variables around a given value.

    Parameters
    ----------
    f_X: 
        Function handle 
    var_model: numpy array 
        Variables of the function f_X 
    eps: double (optional)
        Default value is 1/100 of the variable. 
    compute_all_element: bool 
        If False, only diagonals elements are computed. Default is True. 

    Returns
    ---------
    hess_FD: numpy 2d array
        Gradient of the function f_X with respect to the different variables at var_model

    Joffrey Coheur 19-04-19
    """

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

        if positive_diag: 
            # We force the diag to be positive 
            hess_FD[i, i] = -np.abs(num/(delta_pi**2))
            #print(hess_FD[i, i])

    # 2) Off diagonal terms
    if compute_all_element: 
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



def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)


def computeCorrMat(covMat, sigma): 

    n = covMat.shape[0]
    corrMat = np.zeros((n,n))
    for i in range(n): 
        for j in range(n): 
            corrMat[i,j] = covMat[i,j]/(sigma[i] * sigma[j])

    return corrMat 


def computeCovMat(corrMat, sigma): 

    n = corrMat.shape[0]
    covMat = np.zeros((n,n))
    for i in range(n): 
        for j in range(n): 
            covMat[i,j] = corrMat[i,j] * (sigma[i] * sigma[j])
    return covMat 


def nearPD(A, nit=10):
    """ A function to get the nearest positive definite matrix."""

    sigma = np.sqrt(np.abs(np.diag(A)))
    n = A.shape[0]
    A_ = np.zeros((n, n))
    for i in range(n): 
        for j in range(n): 
            if i == j: 
                A_[i,j] = sigma[i] ** 2 
            else: 
                A_[i,j] = A[i,j]

    corr = computeCorrMat(A_, sigma)
    corr = A 


    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = corr.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)

    Yk = computeCovMat(Yk, sigma)


    return Yk


# def nearPD(A,epsilon=0):
#   """ A function to get the nearest positive definite matrix."""

#     sigma = np.sqrt(np.abs(np.diag(A)))
#     A = computeCorrMat(A, sigma)


#     n = A.shape[0]
#     eigval, eigvec = np.linalg.eig(A)
#     val = np.matrix(np.maximum(eigval,epsilon))
#     vec = np.matrix(eigvec)
#     T = 1/(np.multiply(vec,vec) * val.T)
#     T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
#     B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
#     out = B*B.T

#     out = computeCovMat(out, sigma)

#     return(out)


# def nearPD(A):
# """ A function to get the nearest positive definite matrix."""

#     B = (np.transpose(A) + A)/ 2
#     u, H = polar(B)
#     PSDmat = (B + H)/2 
#     #PSDmat = (PSDmat + np.transpose(PSDmat)) / 2

#     p = 1
#     k = 0
#     while p != 0:

#         try: 
#             R = linalg.cholesky(PSDmat)
#             p = 0
#         except: 
#             print('Non positive defintie Hessian matrix. Set to identity matrix.')
#             p = 1
#             k = k + 1

#         if p != 0:
#             # Ahat failed the chol test. It must have been just a hair off,
#             # due to floating point trash, so it is simplest now just to
#             # tweak by adding a tiny multiple of an identity matrix.
#             eigval, eigvec = np.linalg.eig(PSDmat)
#             mineig = min(eigval)
#             PSDmat = PSDmat + (-mineig*k**2 + np.finfo(float).eps )*np.eye(A.shape[0])

#     PSDmat[0,0] = np.abs(A[0,0])

#     return PSDmat







