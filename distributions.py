import numpy as np 
from scipy import linalg
from scipy import stats

import matplotlib.pyplot as plt


def set_probability_dist(name_list, hyperparam): 
    """ Set the probability distribution. """

    # Check that the function provided in names are implemented 
    implemented_functions = " ".join(["Gaussian", "Uniform", "Gamma", "Mixture"])
    for name in name_list: 
            idx = implemented_functions.find(name)
            if idx < 0: 
                raise ValueError("No {} density function implemented \n"
                "Implemented prior functions are {}.".format(name,implemented_functions))

    if name == 'Gaussian':
        prob_dist = Gaussian(hyperparam)
    elif name == 'Uniform': 
        prob_dist = Uniform(hyperparam)
    elif name == 'Mixture': 
        prob_dist = Mixture(hyperparam)

    return prob_dist



class ProbabilityDistribution: 

    def __init__(self, hyperparam):
        self.hyperparam = hyperparam 
        self.dim = []
        self.distr_support = np.zeros([1,1]) 

    def compute_value(self, X): 
        return 0

    def compute_log_value(self, X):
        return 0

    def compute_density(self):
        """ Compute the probability density function of the associated distribution. 
        Only for one dimension and two dimensions. """

        if self.dim == 1:
            vec_param_i = np.linspace(self.distr_support[0,0],self.distr_support[0,1], 2000)
            delta_param_i = vec_param_i[1]-  vec_param_i[0]
            f_post = np.zeros(vec_param_i.size)
            for i, param_i in np.ndenumerate(vec_param_i): 
                    c_param = np.array([param_i])
                    f_post[i] = self.compute_value(c_param)  

            int_post = np.sum(f_post)*delta_param_i

            plt.figure(200)
            plt.plot(vec_param_i, f_post/int_post)


        elif self.dim == 2:
            vec_param_i = np.linspace(self.distr_support[0,0], self.distr_support[0,1], 50)
            delta_param_i = vec_param_i[1]-  vec_param_i[0]
            vec_param_j = np.linspace(self.distr_support[1,0], self.distr_support[1,1], 50)
            delta_param_j = vec_param_j[1]-  vec_param_j[0]
            f_post = np.zeros(vec_param_i.size, vec_param_j.size)
            for i, param_i in np.ndenumerate(vec_param_i): 
                for j, param_j in np.ndenumerate(vec_param_j): 
                    c_param = np.array([param_i, param_j])
                    f_post[i,j] = self.compute_value(c_param) 

            marginal_post_1 = np.sum(f_post*delta_param_j, axis=1)
            marginal_post_2 = np.sum(f_post*delta_param_i, axis=0)
            plt.figure(1)
            plt.plot(vec_param_i, marginal_post_1)
            plt.figure(2)
            plt.plot(vec_param_j, marginal_post_2)

        return f_post 

    def update_eval(self): 
        """ For Bayesian posterior only """
        pass

    def save_value(self, name_file): 
        """ For Bayesian posterior only """ 
        pass


class Gaussian(ProbabilityDistribution): 

    def __init__(self, hyperparam): 
        ProbabilityDistribution.__init__(self, hyperparam)

        self.mean = np.array([hyperparam[0][0]]) 
        self.cov = np.array([[hyperparam[0][1]]])

        # Support of the distribution in each dimension as mean +- 4 sigma 
        self.dim = len(self.mean)
        self.distr_support = np.zeros([self.dim, 2])
        for i in range(self.dim): 
            sigma_ii = np.sqrt(self.cov[i,i])**2
            # lower bound 
            self.distr_support[i,0] = self.mean - 4 * sigma_ii
            # upper bound 
            self.distr_support[i,1] = self.mean + 4 * sigma_ii

        if len(self.mean) < 2: 
            self.inv_cov = 1 / self.cov**2
        else: 
            self.inv_cov = linalg.inv(self.cov)

    def compute_value(self, X): 

        val = np.exp(-1/2*(X - self.mean)*self.inv_cov*np.transpose((X-self.mean))) 

        return val 

    def compute_log_value(self, X):

        log_val = -1/2*(X - self.mean)*self.inv_cov*np.transpose((X-self.mean))

        return log_val

class Uniform(ProbabilityDistribution):  

    def __init__(self, hyperparam): 
        ProbabilityDistribution.__init__(self, hyperparam)

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
        
    def __init__(self, hyperparam): 
        ProbabilityDistribution.__init__(self, hyperparam)

        self.distr_names = hyperparam[0]
        self.distr_param = hyperparam[1]
        self.n_components = len(self.distr_names)

        self.dim = len(self.distr_names)
        self.distr_support = np.zeros([self.dim, 2])

        # Initialize distributions 
        self.mixture_components = []
        for i in range(self.n_components): 
            c_mixt = set_probability_dist([self.distr_names[i]], self.distr_param[i])
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
