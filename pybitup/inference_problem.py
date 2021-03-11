import numpy as np
import pandas as pd 
import pybitup.metropolis_hastings_algorithms as mha


class Sampler: 
    """Take as argument a probability distribution function Probabilitydistribution and sample it. 
    Can be a BayesianPosterior class
    Iterative algorithms are implemented.
    
    prob_dist :

    algo : dict containing the sampling algorithm specifications """ 

    def __init__(self, IO_fileID, prob_dist, algo):
        self.prob_dist = prob_dist 
        self.algo = algo
        self.IO_fileID = IO_fileID
        self.opt_arg = {}
        av_algo = {}

        # List of available algorithms 
        av_algo['RWMH'] = "random-walk Metropolis-Hastings"
        av_algo['AMH'] = "adaptive Metropolis-Hastings"
        av_algo['DR'] = "delayed-rejection Metropolis-Hastings" 
        av_algo['DRAM'] = "delayed-rejection adative Metropolis-Hastings"
        av_algo['HMC'] = "Hamiltonian Monte Carlo"
        av_algo['ISDE'] = "Ito-SDE"

        # Check that the requested algorithm is implemented
        self.algo_name = self.algo['name']
        if self.algo_name not in av_algo: 
            raise ValueError('Algorithm "{}" unknown.'.format(self.algo_name)) 
        else: 
            print("Running {} algorithm.".format(av_algo[self.algo_name]))

        if self.algo_name == "RWMH" or self.algo_name == "AMH" or self.algo_name == "DR" or self.algo_name == "DRAM": 
            # Proposal function need to be defined for these algorithms
            if self.algo['proposal']['covariance']['type'] == "diag": 
                if isinstance(self.algo['proposal']['covariance']['value'], str):  
                    reader = pd.read_csv(self.algo['proposal']['covariance']['value'], header=None)
                    A = np.transpose(reader.values)
                    self.proposal_cov = np.diag(A[0])
                else:
                    self.proposal_cov = np.diag(self.algo['proposal']['covariance']['value'])
            elif self.algo['proposal']['covariance']['type'] == "full":
                self.proposal_cov = np.array(self.algo['proposal']['covariance']['value']) 
            else: 
                print("Invalid InferenceAlgorithmProposalConvarianceType name {}".format(self.algo['proposal']['covariance']['type']))


        # Number of iterations (must be an integer)
        self.n_iterations = int(self.algo['n_iterations']) 
        
        # Iterations at which the evaluations of the distribution are saved
        # Default value is 10
        if self.algo.get("save_eval_freq") is not None: 
            self.opt_arg['save_eval_freq'] = int(self.algo['save_eval_freq']) 
        else: # Default 
            self.opt_arg['save_eval_freq'] = 10
        
        # Compute the maximum value of the distribution (useful in the case of finding the MAP, but not useful for sampling from know distribution)
        # Default is false 
        self.opt_arg["estimate_max_distr"] = False 
        if self.algo.get("estimate_max_distr") is not None: 
            if self.algo['estimate_max_distr'] == "yes": 
                self.opt_arg["estimate_max_distr"] = True 
             
        # TODO: The sigma in the likelihood (for BayesianPosterior) is also unknown and we estimate his value with the MCMC algo 
        self.opt_arg["estimate_sigma"] = False 
        if self.algo.get("estimate_sigma") is not None:
            if self.algo['estimate_sigma'] == "yes": 
                self.opt_arg["estimate_sigma"] = True 

    def sample(self, sample_init):
        """ Sample the probability density function given in input of the class. 
        We draw sequence of dependent samples from markov chains using iterative algorithms. """

        if self.algo_name == "RWMH": 
            run_MCMCM = mha.MetropolisHastings(self.IO_fileID, self.n_iterations, sample_init, self.proposal_cov, self.prob_dist, self.opt_arg)  
        elif self.algo_name == "AMH": 
            starting_it = int(self.algo['AMH']['starting_it'])
            updating_it = int(self.algo['AMH']['updating_it'])
            eps_v = self.algo['AMH']['eps_v']
            run_MCMCM = mha.AdaptiveMetropolisHastings(self.IO_fileID, self.n_iterations, sample_init, self.proposal_cov, self.prob_dist,  starting_it, updating_it, eps_v, self.opt_arg)
        elif self.algo_name == "DR": 
            gamma = self.algo['DR']['gamma']
            run_MCMCM = mha.DelayedRejectionMetropolisHastings(self.IO_fileID, self.n_iterations, sample_init, self.proposal_cov, self.prob_dist, gamma, self.opt_arg)

        elif self.algo_name == "DRAM": 
            starting_it = int(self.algo['DRAM']['starting_it'])
            updating_it = int(self.algo['DRAM']['updating_it'])
            eps_v = self.algo['DRAM']['eps_v']
            gamma = self.algo['DRAM']['gamma']
            run_MCMCM = mha.DelayedRejectionAdaptiveMetropolisHastings(self.IO_fileID, self.n_iterations, sample_init, self.proposal_cov, self.prob_dist, starting_it, updating_it, eps_v, gamma, self.opt_arg)
        elif self.algo_name == "HMC": 
            C_matrix = self.algo['covariance']
            gradient = self.algo['gradient']
            step_size = self.algo['HMC']['step_size']
            num_steps = self.algo['HMC']['num_steps']
            run_MCMCM = mha.HamiltonianMonteCarlo(self.IO_fileID, self.n_iterations, sample_init, self.prob_dist, C_matrix, gradient, step_size, num_steps, self.opt_arg)
        elif self.algo_name == "ISDE": 
            C_matrix = self.algo['covariance']
            gradient = self.algo['gradient']
            h = self.algo['ISDE']['h']
            f0 = self.algo['ISDE']['f0']
            run_MCMCM = mha.ito_SDE(self.IO_fileID, self.n_iterations,sample_init, self.prob_dist, C_matrix, gradient, h, f0, self.opt_arg)
        
        run_MCMCM.run_algorithm()