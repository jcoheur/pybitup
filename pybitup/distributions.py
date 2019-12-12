import numpy as np 
from scipy import linalg
from scipy import stats
from scipy.integrate import simps

import matplotlib.pyplot as plt


def set_probability_dist(name_list, hyperparam, n_rand_var): 
    """ Set the probability distribution. """

    # Check that the function provided in names are implemented 
    implemented_functions = " ".join(["Gaussian", "Uniform", "Gamma", "Mixture"])
    for name in name_list: 
            idx = implemented_functions.find(name)
            if idx < 0: 
                raise ValueError("No {} density function implemented \n"
                "Implemented prior functions are {}.".format(name,implemented_functions))

    if name == 'Gaussian':
        prob_dist = Gaussian(hyperparam, n_rand_var)
    elif name == 'Uniform': 
        prob_dist = Uniform(hyperparam, n_rand_var)
    elif name == 'Mixture': 
        prob_dist = Mixture(hyperparam, n_rand_var)

    return prob_dist



class ProbabilityDistribution: 

    def __init__(self, hyperparam, n_rand_var):
        self.hyperparam = hyperparam 
        self.dim = n_rand_var
        self.distr_support = np.zeros([1,1]) 

        self.name_random_var = []
        for i in range(self.dim):
            self.name_random_var.append("X{} ".format(i))


    def compute_value(self, X): 
        return 0

    def compute_log_value(self, X):
        return 0

    def compute_density(self, distr_support=[]):
        """ Compute the probability density function of the associated distribution. 
        Only for one dimension and two dimensions. """

        if distr_support: 
            # If we provide a value for the support of the distribution as input 
            self.distr_support = np.array(distr_support)
            # Otherwise it is already implemented, see Gaussian 

        if self.dim == 1:
            vec_param_i = np.linspace(self.distr_support[0,0],self.distr_support[0,1], 2000)
            delta_param_i = vec_param_i[1] - vec_param_i[0]
            f_post = np.zeros(vec_param_i.size)
            for i, param_i in np.ndenumerate(vec_param_i): 
                    c_param = np.array([param_i])
                    f_post[i] = self.compute_value(c_param)  

            #int_post = np.sum(f_post)*delta_param_i
            int_post = simps(f_post, vec_param_i)

            plt.figure(200)
            plt.plot(vec_param_i, f_post/int_post)


        elif self.dim == 2:
            vec_param_i = np.linspace(self.distr_support[0,0], self.distr_support[0,1], 100)
            delta_param_i = np.exp(vec_param_i[1] - vec_param_i[0])
            vec_param_j = np.linspace(self.distr_support[1,0], self.distr_support[1,1], 100)
            delta_param_j = np.exp(vec_param_j[1] - vec_param_j[0])
            f_post = np.zeros([vec_param_i.size, vec_param_j.size])
            for i, param_i in np.ndenumerate(vec_param_i): 
                for j, param_j in np.ndenumerate(vec_param_j): 
                    c_param = np.array([param_i, param_j])
                    f_post[i,j] = self.compute_value(c_param) 

            marginal_post_1 = np.sum(f_post*delta_param_j, axis=1)
            int_f_post  = np.sum(marginal_post_1*delta_param_i, axis=0)
            norm_f_post = f_post / int_f_post

            marginal_post_norm_1 = np.sum(norm_f_post*delta_param_j, axis=1)
            marginal_post_norm_2 = np.sum(norm_f_post*delta_param_i, axis=0)
            plt.figure(200)
            plt.plot(np.exp(vec_param_i), marginal_post_norm_1)
            plt.figure(201)
            plt.plot(np.exp(vec_param_j), marginal_post_norm_2)

        return f_post 


    def save_sample(self, IO_fileID, value): 
        """ Save the sample vaue in a text file""" 

        # Write the Markov chain file 
        IO_fileID['MChains'].write("{}\n".format(str(value).replace('\n', '')))
        np.savetxt(IO_fileID['MChains_csv'], np.array([value]), fmt="%f", delimiter=",")
        
    def update_eval(self): 
        """ For Bayesian posterior only """
        pass

    def save_value(self, name_file): 
        """ For Bayesian posterior only """ 
        pass




class Gaussian(ProbabilityDistribution): 

    def __init__(self, hyperparam, n_rand_var): 
        ProbabilityDistribution.__init__(self, hyperparam, n_rand_var)

        self.mean = np.array(hyperparam[0]).T
        self.cov = np.array(hyperparam[1])

        # Support of the distribution in each dimension as mean +- 4 sigma 
        self.distr_support = np.zeros([self.dim, 2])
        for i in range(self.dim): 
            var_ii = self.cov[i,i]
            # lower bound 
            self.distr_support[i,0] = self.mean[0][i] - 4 * np.sqrt(var_ii)
            # upper bound 
            self.distr_support[i,1] = self.mean[0][i] + 4 * np.sqrt(var_ii)

        if self.dim < 2: 
            self.inv_cov = 1 / self.cov
            det_cov = self.cov
        else: 
            self.inv_cov = linalg.inv(self.cov)
            det_cov = linalg.det(self.cov)

        self.gauss_coeff = 1/np.sqrt((2 * np.pi) ** self.dim * det_cov)
        self.log_gauss_coeff = np.log(self.gauss_coeff)
        

    def compute_exp_arg(self, X): 
       
        diff_x = (X-self.mean[0][:]) 
        M1 = np.matmul(self.inv_cov, np.transpose(diff_x))
        M2 = np.matmul(diff_x, M1) 


        return M2

    def compute_value(self, X): 

        exp_arg = self.compute_exp_arg(X)
        val = self.gauss_coeff*np.exp(-1/2*exp_arg) 

        return val 

    def compute_log_value(self, X):

        exp_arg = self.compute_exp_arg(X)   
        log_val = self.log_gauss_coeff - 1/2*exp_arg

        return log_val

        # rv = stats.multivariate_normal(self.mean[0][:], self.cov)
        # return rv.logpdf(X)


class Uniform(ProbabilityDistribution):  

    def __init__(self, hyperparam, n_rand_var): 
        ProbabilityDistribution.__init__(self, hyperparam, n_rand_var)

        self.lb = hyperparam[0]
        self.ub = hyperparam[1]
        self.prob = 1/abs(hyperparam[1] - hyperparam[0])

        self.distr_support = np.array([self.lb, self.ub])

    def compute_value(self, X):

        if X < self.lb or X > self.ub:
            prob = 0 
        else: 
            prob = self.prob 

        return prob 


class Mixture(ProbabilityDistribution):
        
    def __init__(self, hyperparam, n_rand_var): 
        ProbabilityDistribution.__init__(self, hyperparam, n_rand_var)

        self.distr_names = hyperparam[0]
        self.distr_param = hyperparam[1]
        self.n_components = len(self.distr_names)

        self.dim = len(self.distr_names)
        self.distr_support = np.zeros([self.dim, 2])

        # Initialize distributions 
        self.mixture_components = []
        for i in range(self.n_components): 
            c_mixt = set_probability_dist([self.distr_names[i]], self.distr_param[i], n_rand_var)
            self.mixture_components.append(c_mixt)

            self.distr_support[i,0] = c_mixt.distr_support[0]
            self.distr_support[i,1] = c_mixt.distr_support[1]


   
    def compute_value(self, X): 
        """ compute_value compute the value of the joint pdf at X, where 
        X is a (1 x n_param) numpy array. We sample from known distribution"""

        Y = 1

        for i in range(self.n_components): 
            c_mixt = self.mixture_components[i]
            Y *= c_mixt.compute_value(X[i])

        return Y 

    def compute_log_value(self, X): 

        Y = self.compute_value(X)

        return np.log(Y)
