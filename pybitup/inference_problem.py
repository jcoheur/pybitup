import numpy as np

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

        # List of implemented algorithms 
        av_algo = ['RWMH', 'AMH', 'DR', 'DRAM', 'Ito-SDE']

        # Check that the requested algorithm is implemented
        self.algo_name = self.algo['name']
        if self.algo_name not in av_algo: 
            raise ValueError('Algorithm "{}" unknown.'.format(self.algo_name)) 

        if self.algo_name == "RWMH" or self.algo_name == "AMH" or self.algo_name == "DR" or self.algo_name == "DRAM": 
            # Proposal function need to be defined for these algorithms
            if self.algo['proposal']['covariance']['type'] == "diag": 
                self.proposal_cov = np.diag(self.algo['proposal']['covariance']['value'])
            elif self.algo['proposal']['covariance']['type'] == "full":
                self.proposal_cov = np.array(self.algo['proposal']['covariance']['value']) 
            else: 
                print("Invalid InferenceAlgorithmProposalConvarianceType name {}".format(self.algo['proposal']['covariance']['type']))


        # Number of iterations (must be an integer)
        self.n_iterations = int(self.algo['n_iterations']) 

    def sample(self, sample_init):
        """ Sample the probability density function given in input of the class. 
        We draw sequence of dependent samples from markov chains using iterative algorithms. """

        if self.algo_name == "RWMH": 
            print("Using random-walk Metropolis-Hastings algorithm.")
            run_MCMCM = mha.MetropolisHastings(self.IO_fileID, "sampling", self.n_iterations, sample_init, self.proposal_cov, self.prob_dist)  
        elif self.algo_name == "AMH": 
            print("Using adaptive random-walk Metropolis-Hastings algorithm.")
            starting_it = int(self.algo['AMH']['starting_it'])
            updating_it = int(self.algo['AMH']['updating_it'])
            eps_v = self.algo['AMH']['eps_v']
            run_MCMCM = mha.AdaptiveMetropolisHastings(self.IO_fileID, "sampling", self.n_iterations, sample_init, self.proposal_cov, self.prob_dist,
                                                        starting_it, updating_it, eps_v)
        elif self.algo_name == "DR": 
            print("Using delayed-rejection random-walk Metropolis-Hastings algorithm.")
            gamma = self.algo['DR']['gamma']
            run_MCMCM = mha.DelayedRejectionMetropolisHastings(self.IO_fileID, "sampling", self.n_iterations, sample_init, self.proposal_cov, self.prob_dist,
                                                        gamma)

        elif self.algo_name == "DRAM": 
            print("Using delayed-rejection adaptive random-walk Metropolis-Hastings algorithm.")
            starting_it = int(self.algo['DRAM']['starting_it'])
            updating_it = int(self.algo['DRAM']['updating_it'])
            eps_v = self.algo['DRAM']['eps_v']
            gamma = self.algo['DRAM']['gamma']
            run_MCMCM = mha.DelayedRejectionAdaptiveMetropolisHastings(self.IO_fileID, "sampling", self.n_iterations, sample_init, self.proposal_cov, self.prob_dist,
                                                                        starting_it, updating_it, eps_v, gamma)
        elif self.algo_name == "Ito-SDE": 
            print("Running Ito-SDE algorithm.")
            h = self.algo['Ito-SDE']['h']
            f0 = self.algo['Ito-SDE']['f0']
            run_MCMCM = mha.ito_SDE(self.IO_fileID, "sampling", self.n_iterations, sample_init, self.prob_dist,
                                    h, f0)
        
        run_MCMCM.run_algorithm()