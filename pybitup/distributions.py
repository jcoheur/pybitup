import pathlib
import numpy as np
from scipy import linalg, stats, special
from scipy.integrate import simps

import matplotlib.pyplot as plt


def set_probability_dist(name_list, hyperparam, n_rand_var): 
    """ Set the probability distribution. """

    IMPLEMENTED_PDFS = ["Gaussian", 
                        "Uniform", 
                        "Gamma", 
                        "Mixture", 
                        "SoizePDF"]


    # Check that the function provided in names are implemented 
    implemented_functions = " ".join(IMPLEMENTED_PDFS)
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
    elif name == 'SoizePDF':
        prob_dist = SoizePDF(hyperparam, n_rand_var=2)
        # No hyperparam and dimension fixed to two 

    return prob_dist



class ProbabilityDistribution: 
    """
    A generic class for probability density functions. 
    
    Attributes
    ----------
    hyperparam : list
        List that contains the different distribution hyperparameters. 
    n_rand_var : char 
        Number of random variable of the distribution. 
   


    Methods
    -------------
    compute_value(self, X)
        Computes the value of the probability density at X. 
    compute_log_value(self, X):
        Computes the log-value of the probability density at X. 
    compute_density(self, distr_support=[])
        Computes the probability density along the support. 
    save_sample(self, IO_fileID, value): 
        Saves the sample value in a file. 
    update_eval(self):
        Updates function evaluation in Bayesian posterior dsitribution. 
    save_value(self, name_file): 
        Saves the value of the function evaluation in Bayesian posterior dsitribution. 
   
    """

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
        """ Compute the probability density function of the associated distribution along the . 
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
                    f_post[i] = self.compute_value_no_reparam(c_param)  

            #int_post = np.sum(f_post)*delta_param_i
            int_post = simps(f_post, vec_param_i)
            norm_f_post = f_post / int_post

            plt.figure(200)
            plt.plot(vec_param_i, norm_f_post)

            x = np.linspace(stats.norm.ppf(0.01, loc=113000, scale=1000), stats.norm.ppf(0.99, loc=113000, scale=1000), 100)
            plt.plot(x, stats.norm.pdf(x, loc=113000, scale=1000), 'r-', lw=2, alpha=1.0, label='norm pdf')

        elif self.dim == 2:
            n_points_1d = 400 # 200

            # # The support is uniform in the reparameterized space 
            vec_param_i = np.exp(np.linspace(self.distr_support[0,0], self.distr_support[0,1], n_points_1d))
            vec_param_j = np.exp(np.linspace(self.distr_support[1,0], self.distr_support[1,1], n_points_1d))

            # The support is uniform in the original parameter space 
            # vec_param_i = np.linspace(np.exp(self.distr_support[0,0]), np.exp(self.distr_support[0,1]), n_points_1d)
            # vec_param_j = np.linspace(np.exp(self.distr_support[1,0]), np.exp(self.distr_support[1,1]), n_points_1d)

            # Original support 
            # vec_param_i = np.linspace(self.distr_support[0,0], self.distr_support[0,1], n_points_1d)
            # vec_param_j = np.linspace(self.distr_support[1,0], self.distr_support[1,1], n_points_1d)


            delta_param_i = vec_param_i[1] - vec_param_i[0]
            delta_param_j = vec_param_j[1] - vec_param_j[0]
            f_post = np.zeros([vec_param_i.size, vec_param_j.size])
            for i, param_i in np.ndenumerate(vec_param_i): 
                for j, param_j in np.ndenumerate(vec_param_j): 
                    c_param = np.array([param_i, param_j])
                    f_post[i,j] = self.compute_value_no_reparam(c_param) 

            # marginal_post_1 = np.sum(f_post*delta_param_j, axis=1)
            # int_f_post  = np.sum(marginal_post_1*delta_param_i, axis=0)
            # norm_f_post = f_post / int_f_post

            # marginal_post_norm_1 = np.sum(norm_f_post*delta_param_j, axis=1)
            # marginal_post_norm_2 = np.sum(norm_f_post*delta_param_i, axis=0)
            # plt.figure(200)
            # plt.plot(vec_param_i, marginal_post_norm_1)
            # plt.figure(201)
            # plt.plot(vec_param_j, marginal_post_norm_2)

        post_num_eval_file_path = pathlib.Path("output", "posterior_numerical_evaluation.npz")
        np.savez(post_num_eval_file_path, x=vec_param_i, y=vec_param_j, z=f_post)

        return f_post 


    def save_sample(self, IO_fileID, value): 
        """ Save the sample value in a csv file. """ 
        
        # Write the Markov chain file 
        np.savetxt(IO_fileID['MChains'], np.array([value]), fmt="%f", delimiter=",")
        
    def compute_value_no_reparam(self, X):
        """ For Bayesian posterior, this function returns the
        density evaluation with the initial parameterization.
        This is used for the direct evaluation of the density (see 
        compute_density in the same class). 

        For other densities, this is the same as self.compute_value(X).
        """

        return self.compute_value(X) 

    def update_eval(self): 
        """ For Bayesian posterior only. 
        We need to define it in the master class for generality of the sampling methods."""
        pass

    def save_value(self, IO_util, current_it): 
        """ For Bayesian posterior only.
        We need to define it in the master class for generality of the sampling methods."""
        pass




class Gaussian(ProbabilityDistribution): 
    """
    A class for computing the density of a Gaussian distribution function. 
    
    Attributes
    ----------
    No new attributes.    


    Methods
    -------------

    """

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

        #return log_val

        rv = stats.multivariate_normal(self.mean[0][:], self.cov)
        return rv.logpdf(X)

    def compute_grad_log_value(self, X):

        return 1

    def invcdf(self,x):

        [a,b] = [self.mean,self.cov]
        return  a+np.sqrt(2)*b*special.erfinv(2*np.array(x)-1)

    def coef(self,nbrCoef):

        [a,b] = [self.mean,self.cov]
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0].fill(a)
        coef[1] = b**2*n
        return coef



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

    def invcdf(self,x):

        [a,b] = [self.lb,self.ub]
        return (b-a)*np.array(x)+a

    def coef(self,nbrCoef):

        [a,b] = [self.lb,self.ub]
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0].fill((b+a)/2)
        coef[1] = ((b-a)*n/2)**2/(4*n**2-1)
        return coef



class Exponential(ProbabilityDistribution):

    def __init__(self, hyperparam, n_rand_var): 
        ProbabilityDistribution.__init__(self, hyperparam, n_rand_var)

        self.lamb = np.array(hyperparam)

    def coef(self,nbrCoef):

        a = self.lamb
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = a*(1+2*n)
        coef[1] = (a*n)**2
        return coef

    def invcdf(self,x):

        a = self.lamb
        return -np.log(1-np.array(x))/a

    def compute_value(self,x):

        a = self.lamb
        return a*np.exp(-a*np.array(x))


class Gamma(ProbabilityDistribution):

    def __init__(self, hyperparam, n_rand_var): 
        ProbabilityDistribution.__init__(self, hyperparam, n_rand_var)

        self.k = np.array(hyperparam[0])
        self.theta = np.array(hyperparam[1])

    def coef(self,nbrCoef):

        [a,b] = [self.k,self.theta]
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = (2*n+a)*b
        coef[1] = (n+a-1)*n*b**2
        return coef

    def invcdf(self, x):

        [a,b] = [self.k,self.theta]
        return b*special.gammaincinv(a,np.array(x))


    def compute_value(self,x):

        [a,b] = [self.k,self.theta]
        return x**(a-1)*np.exp(-np.array(x)/b)/(special.gamma(a)*b**a)



class Lognormal(ProbabilityDistribution):

    def __init__(self, hyperparam, n_rand_var): 
        ProbabilityDistribution.__init__(self, hyperparam, n_rand_var)

        self.mean = np.array(hyperparam[0]).T
        self.cov = np.array(hyperparam[1])

    def coef(self,nbrCoef):

        [a,b] = [self.mean,self.cov]
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))
        coef[0] = (np.exp((n+1)*b**2)+np.exp(n*b**2)-1)*np.exp(((2*n-1)*b**2)/2+a)
        coef[1] = (np.exp(n*b**2)-1)*np.exp((3*n-2)*b**2+2*a)
        return coef

    def invcdf(self,x):

        [a,b] = [self.mean,self.cov]
        return np.exp(a+np.sqrt(2)*b*special.erfinv(2*np.array(x)-1))

    def compute_value(self,x):

        [a,b] = [self.mean,self.cov]
        return np.exp(-0.5*((np.log(np.array(x))-a)/b)**2)/(np.array(x)*b*np.sqrt(2*np.pi))


class Beta(ProbabilityDistribution):

    def __init__(self, hyperparam, n_rand_var): 
        ProbabilityDistribution.__init__(self, hyperparam, n_rand_var)

        self.alpha = np.array(hyperparam[0])
        self.beta = np.array(hyperparam[1])

    def coef(self,nbrCoef):

        [a,b] = [self.alpha,self.beta]
        n = np.arange(nbrCoef)
        coef = np.zeros((2,nbrCoef))

        nab = 2*n+a+b
        B1 = a*b*1./((a+b+1)*(a+b)**2)
        B2 = (n+a-1)*(n+b-1)*n*(n+a+b-2)/((nab-1)*(nab-3)*(nab-2)**2+2*((n==0)+(n==1)))
        coef[0] = ((a-1)**2-(b-1)**2)*0.5/(nab*(nab-2)+(nab==0)+(nab==2))+0.5
        coef[1] = np.where((n==0)+(n==1),B1,B2)
        return coef

    def invcdf(self,x):

        [a,b] = [self.alpha,self.beta]
        return special.betaincinv(a,b,np.array(x))

    def compute_value(self,x):

        [a,b] = [self.alpha,self.beta]
        return x**(a-1)*(1-np.array(x))**(b-1)/special.beta(a,b)


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
        """ Take the log of the density distribution of a mixture of other 
        distribution. 

        !!!! BE CAREFUL WITH THIS !!!! 
        We are tempted to implement :

            Y = self.compute_value(X)
            return np.log(Y)
        
        Don't! With uniform distribution with large support,
        the probability density is very low. The product can be then
        assimilated to zero by python. That's why actually we use the log instead. But do not take simply the log of the function 
        "compute_value"! In which case, you will take log(0.0). Thus, we need to reimplement the same function. The resulting log will be finite. 

        """

        Y = 0
        for i in range(self.n_components): 
            c_mixt = self.mixture_components[i]
            # Log (ab) = log(a) + log(b) 
            Y += np.log(c_mixt.compute_value(X[i]))
        return Y



class SoizePDF(ProbabilityDistribution): 
    """ Test case PDF for comparing RMWH, ito-SDE, etc. 
    From Soize 2017, Uncertainty Quantification, An accelerated Course ... pp. 65 """

    def __init__(self, hyperparam, n_rand_var): 
        ProbabilityDistribution.__init__(self, hyperparam, n_rand_var)

        self.c0 = 1 

    def compute_value(self, X): 

        val = self.c0 * np.exp( -15 * (X[0]**3 - X[1])**2 - (X[1] - 0.3)**4) 

        return val 

    def compute_log_value(self, X):
    
        log_val = np.log(self.c0) - 15 * (X[0]**3 - X[1])**2 - (X[1] - 0.3)**4

        return log_val 

    def compute_grad_log_value(self, X):

        return 1


class Joint:
    """Class of joint probability distribution function"""

    def __init__(self,dist): self.dist = np.copy(np.atleast_1d(dist))
    def __setitem__(self,i,dist): self.dist[i] = dist
    def __getitem__(self,i): return self.dist[i]

    def compute_value(self,point):

        dim = self.dist.shape[0]
        point = np.atleast_2d(point)
        resp = [self.dist[i].compute_value(point[:,i]) for i in range(dim)]
        resp = np.squeeze(np.prod(resp,axis=0))
        return resp