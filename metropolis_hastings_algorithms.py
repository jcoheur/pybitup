import os
import numpy as np
import random
from scipy import linalg
import time


def bayes_formula():
    """ Compute the posterior distribution using Bayes formula """
    a = 1


class MetropolisHastings:

    def __init__(self, caseName, nIterations, param_init, V, model, prior, data, f_X):
        self.caseName = caseName
        self.nIterations = nIterations
        self.param_init = param_init
        self.V = V
        self.model = model
        self.prior = prior
        self.data = data
        self.f_X = f_X

        # Initialize parameters and functions
        self.n_param = self.param_init.size
        self.current_val = self.param_init
        self.new_val = np.zeros(self.n_param)
        self.current_fun_eval = f_X(self.current_val)
        self.SS_current_fun_eval = sum_of_square(self.data, self.current_fun_eval)
        # Store the maximum of the log-likelihood function
        self.max_LL = self.SS_current_fun_eval
        self.arg_max_LL = self.current_val

        # Reparametrization
        self.x_parametrization = self.model.parametrization_forward
        self.y_parametrization = self.model.parametrization_backward
        self.det_jac = self.model.parametrization_det_jac
        self.vec_X_parametrized = np.zeros(self.n_param)
        self.Y_parametrized = np.zeros(self.n_param)

        # Save the initial guesses in output folder
        os.system("mkdir output")
        # Create the output file mcmc_chain.dat and close it
        tmp_fileID = open("output/mcmc_chain.dat", "w")
        tmp_fileID.close()
        # Re-open it in read and write mode (option r+ cannot create non existing file)
        self.fileID = open("output/mcmc_chain.dat", "r+")
        np.save("output/fun_eval.0", self.current_fun_eval)
        self.write_val(self.current_val)

        # Cholesky decomposition of V. Constant if MCMC is not adaptive
        self.R = linalg.cholesky(V)

        # Monitoring the chain
        self.n_rejected = 0

        # Print current time and start clock count
        print("Start time {}" .format(time.asctime(time.localtime())))
        self.t1 = time.clock()

    def random_walk_loop(self):

        for i in range(self.nIterations+1):

            self.compute_new_val()
            self.compute_acceptance_ratio()
            self.accept_reject()

            # We save 100 function evaluation for the post process
            self.write_fun_eval(i, self.nIterations/100, self.current_fun_eval)

            # We estimate time after a hundred iterations
            if i == 100:
                self.compute_time(self.t1)

            # Save the next current value
            self.write_val(self.current_val)

        self.compute_covariance()

        self.terminate_loop()

    def compute_new_val(self):

        # Guess parameter (candidate or proposal)
        z_k = self.compute_multivariate_normal()

        ## Without parameterization
        #self.new_val[:] = self.current_val[:] + np.transpose(np.matmul(self.R, np.transpose(z_k)))

        self.vec_X_parametrized[:] = self.x_parametrization(
            self.current_val, self.model.P)
        self.Y_parametrized[:] = self.vec_X_parametrized[:] + \
            np.transpose(np.matmul(self.R, np.transpose(z_k)))
        self.new_val[:] = self.y_parametrization(
            self.Y_parametrized[:], self.model.P)

    def compute_acceptance_ratio(self):
        # Compute ratio of prior distributions
        pi_0_X = self.prior.compute_value(self.current_val[:])
        pi_0_Y = self.prior.compute_value(self.new_val[:])

        # Test prior values to avoid computation of 0/0
        if pi_0_Y <= 0:
            # A new sample out of bounds always rejected
            self.r = 0

        elif pi_0_X <= 0:
            # Previous sample out of bounds always make the new one accepted
            # (if it is in the bounds, otherwise it is in the case above)
            self.r = 1

        else:
            # Acceptance ratio
            self.new_fun_eval = self.f_X(self.new_val)
            self.SS_new_fun_eval = sum_of_square(self.data, self.new_fun_eval)

            # Compare value of SS_Y to get MLE
            if abs(self.SS_new_fun_eval) < abs(self.max_LL):
                self.max_LL = self.SS_new_fun_eval
                self.arg_max_LL = self.new_val[:]

            # Ratio of the determinant of jacobians
            r_det_jac = self.det_jac(self.new_val[:]) / self.det_jac(self.current_val[:])
            self.r = np.exp(self.SS_new_fun_eval-self.SS_current_fun_eval) * r_det_jac

            # Multiply by the ratio of prior values
            r_pi_0 = pi_0_Y / pi_0_X  # This ratio can be compute safely
            self.r *= r_pi_0

        self.alpha = min(1, self.r)

    def accept_reject(self):

        # Update
        u = random.random()  # Uniformly distributed number in the interval [0,1)
        if u < self.alpha:  # Accepted
            self.current_val[:] = self.new_val[:]
            self.current_fun_eval = self.new_fun_eval
            self.SS_current_fun_eval = self.SS_new_fun_eval
        else:  # Rejected, current val remains the same
            self.n_rejected += 1

    def compute_covariance(self):

        print("Initial covariance matrix is :")
        print(self.V)

        # Load all the previous iterations
        param_values = np.zeros((self.nIterations+2, self.n_param))
        j = 0
        self.fileID.seek(0)
        for line in self.fileID:
            c_chain = line.strip()
            param_values[j, :] = np.fromstring(c_chain[1:len(c_chain)-1], sep=' ')
            j += 1

        # Compute sample mean and covariance
        mean_c = np.mean(param_values, axis=0)
        cov_c = np.cov(param_values, rowvar=False)
        print("Final chain Covariance is")
        print(cov_c)

    def compute_multivariate_normal(self):

        mv_norm = np.zeros(self.n_param)
        for j in range(self.n_param):
            mv_norm[j] = random.gauss(0, 1)

        return mv_norm

    def terminate_loop(self):
        self.fileID.close()
        print("End time {}" .format(time.asctime(time.localtime())))
        print("Elapsed time: {} sec".format(time.strftime(
            "%H:%M:%S", time.gmtime(time.clock()-self.t1))))
        print("Rejection rate is {} %".format(self.n_rejected/self.nIterations*100))

        with open("output/output.dat", 'w') as output_file:
            output_file.write("Rejection rate is {} % \n".format(
                self.n_rejected/self.nIterations*100))
            output_file.write("Maximum Likelihood Estimator (MLE) \n")
            output_file.write("{} \n".format(self.arg_max_LL))
            output_file.write("Log-likelihood value \n")
            output_file.write("{}".format(self.max_LL))

    def compute_time(self, t1):
        """ Return the time in H:M:S from time t1 to current clock time """

        print("Estimated time: {}".format(time.strftime("%H:%M:%S",
                                                time.gmtime((time.clock()-t1) / 100.0 * self.nIterations))))

    def write_fun_eval(self, current_it, save_freq, fun_val):
        if current_it % (save_freq) == 0:
            np.save("output/fun_eval.{}".format(current_it), fun_val)

    def write_val(self, value):
        # Write the new current val parameter values
        self.fileID.write("{}\n".format(str(value).replace('\n', '')))
        # replace is used to remove the end of lines in the arrays


class AdaptiveMetropolisHastings(MetropolisHastings):

    def __init__(self, caseName, nIterations, param_init, V, model, prior, data, f_X,
            starting_it, updating_it, eps_v):
        MetropolisHastings.__init__(
            self, caseName, nIterations, param_init, V, model, prior, data, f_X)
        self.starting_it = starting_it
        self.updating_it = updating_it
        self.S_d = 2.38**2/self.n_param
        self.eps_Id = eps_v*np.eye(self.n_param)

    def random_walk_loop(self):

        for i in range(self.nIterations+1):

            self.compute_new_val()
            self.compute_acceptance_ratio()
            self.accept_reject()
            self.adapt_covariance(i)

            # We save 100 function evaluation for the post process
            self.write_fun_eval(i, self.nIterations/100, self.current_fun_eval)

            # We estimate time after a hundred iterations
            if i == 100:
                self.compute_time(self.t1)

            # Save the next current value
            self.write_val(self.current_val)

        self.compute_covariance()

        self.terminate_loop()

    def adapt_covariance(self, i):

        if i >= self.starting_it:
            # Initialisation
            if i == self.starting_it:
                # Load all the previous iterations
                param_values = np.zeros((self.starting_it+1, self.n_param))
                j = 0
                self.fileID.seek(0)
                for line in self.fileID:
                    c_chain = line.strip()
                    param_values[j, :] = np.fromstring(c_chain[1:len(c_chain)-1], sep=' ')
                    j += 1

                # Compute current sample mean and covariance
                self.X_av_i = np.mean(param_values, axis=0)
                self.V_i = self.S_d*(np.cov(param_values, rowvar=False) + self.eps_Id)

            X_i = self.current_val

            # Recursion formula to compute the mean based on previous value
            X_av_ip = (1/(i+2))*((i+1)*self.X_av_i + X_i)

            # Recursion formula to compute the covariance V (Haario, Saksman, Tamminen, 2001)
            V_ip = (i/(i+1))*self.V_i + (self.S_d/(i+1))*(self.eps_Id + (i+1)*(np.tensordot(np.transpose(self.X_av_i), self.X_av_i, axes=0)) -
                                                (i+2)*(np.tensordot(np.transpose(X_av_ip), X_av_ip, axes=0)) + np.tensordot(np.transpose(X_i), X_i, axes=0))

            # Update mean and covariance
            self.V_i = V_ip
            self.X_av_i = X_av_ip

            # The new value for the covariance is updated only every updating_it iterations
            if i % (self.updating_it) == 0:
                self.update_covariance()

    def update_covariance(self):
        self.R = linalg.cholesky(self.V_i)


class DelayedRejectionMetropolisHastings(MetropolisHastings):

    def __init__(self, caseName, nIterations, param_init, V, model, prior, data, f_X,
            gamma):
        MetropolisHastings.__init__(
            self, caseName, nIterations, param_init, V, model, prior, data, f_X)
        self.gamma = gamma

        # Compute inverse of covariance
        inv_R = linalg.inv(self.R)
        self.inv_V = inv_R*np.transpose(inv_R)

    def accept_reject(self):

        # Update
        u = random.random()  # Uniformly distributed number in the interval [0,1)
        if u < self.alpha:  # Accepted
            self.current_val[:] = self.new_val[:]
            self.current_fun_eval = self.new_fun_eval
            self.SS_current_fun_eval = self.SS_new_fun_eval
        else:  # Delayed rejection
            self.delayed_rejection()

    def delayed_rejection(self):

        # Delayed rejection algorithm

        # New Guess parameter (candidate or proposal)
        z_k = self.compute_multivariate_normal()

        self.Y_parametrized[:] = self.vec_X_parametrized[:] + \
            np.transpose(self.gamma*np.matmul(self.R, np.transpose(z_k)))
        DR_new_val = self.y_parametrization(self.Y_parametrized[:], self.model.P)

        # Compute ratio of prior distributions
        pi_0_X = self.prior.compute_value(self.current_val[:])
        pi_0_Y = self.prior.compute_value(DR_new_val[:])

        # Test prior values to avoid computation of 0/0
        if pi_0_Y <= 0:
            # A new sample out of bounds always rejected
            r_2 = 0

        elif pi_0_X <= 0:
            # Previous sample out of bounds always make the new one accepted
            # (if it is in the bounds, otherwise it is in the case above)
            r_2 = 1

        else:
            # Acceptance ratio
            DR_new_fun_eval = self.f_X(DR_new_val)
            SS_Y_2 = sum_of_square(self.data, DR_new_fun_eval)
            r_12 = np.exp(self.SS_new_fun_eval - SS_Y_2)
            alpha_12 = min(1, r_12)
            diff_estimates = self.current_val - DR_new_val
            M1 = np.matmul(diff_estimates, self.inv_V)
            M2 = np.matmul(M1, np.transpose(diff_estimates))
            r_2 = np.exp(SS_Y_2 - self.SS_current_fun_eval) * \
                            np.exp(-1/2*M2) * (1 - alpha_12) / (1 - self.alpha)
            #print(alpha_12, self.alpha )
            alpha_2 = min(1, r_2)

            # Update 2
            u = random.random()  # uniformly distributed number in the interval [0,1]
            if u < alpha_2:  # Accepted
                self.current_val[:] = DR_new_val[:]
                self.current_fun_eval = DR_new_fun_eval
                self.SS_current_fun_eval = SS_Y_2
            else:  # Rejected, current val remains the same
                self.n_rejected += 1


class DelayedRejectionAdaptiveMetropolisHastings(AdaptiveMetropolisHastings, DelayedRejectionMetropolisHastings):

    def __init__(self, caseName, nIterations, param_init, V, model, prior, data, f_X,
            starting_it, updating_it, eps_v, gamma):
        # There is still a problem here as we initialize two times the mother class MetropolisHastings, while once is enough. Don't know yet how to do.
        AdaptiveMetropolisHastings.__init__(
            self, caseName, nIterations, param_init, V, model, prior, data, f_X, starting_it, updating_it, eps_v)
        DelayedRejectionMetropolisHastings.__init__(
            self, caseName, nIterations, param_init, V, model, prior, data, f_X, gamma)

    def update_covariance(self):
        self.R = linalg.cholesky(self.V_i)
        # The inverse of the covariance must be update accordingly for the DRAM
        inv_R = linalg.inv(self.R)
        self.inv_V = inv_R*np.transpose(inv_R)


class ito_SDE(MetropolisHastings):
    """Implementation of the Ito-SDE (Arnst et al.) mcmc methods. Joffrey Coheur 17-04-19."""

    def __init__(self, caseName, nIterations, param_init, V, model, prior, data, f_X):
        MetropolisHastings.__init__(
            self, caseName, nIterations, param_init, V, model, prior, data, f_X)

        # Set forward and backward change of variables
        self.cv_forward = self.x_parametrization
        self.cv_backward = self.y_parametrization

        # Change variable of initial parameters
        xi = self.cv_forward(self.param_init)

        # Definition of the log-likelihood function
        self.SS_X = lambda x: sum_of_square(data, self.f_X(self.cv_backward(x)))
        #self.log_prior = lambda cv_xi : np.log(np.exp(cv_xi[0]) * np.exp(cv_xi[1]) * 1/((sec(1/2 + np.arctan(cv_xi[2])/np.pi))**2))

        # Compute hessian matrix for scaling
        hess_FD = -computeHessianFD(self.SS_X, xi, eps=0.0000001)
        self.C_approx = linalg.inv(hess_FD)  # np.array([np.log(1e4)**2])
        L_c = linalg.cholesky(self.C_approx)
        self.inv_L_c_T = linalg.inv(np.transpose(L_c))

        # Parameters for the ito-sde resolution
        self.h = 0.5
        self.f0 = 4
        self.hfm = 1 - self.h*self.f0/4
        self.hfp = 1 + self.h*self.f0/4

        # Initial parameters
        self.xi_nm = np.array(xi)
        self.P_nm = np.zeros(self.n_param)

    def random_walk_loop(self):

        for i in range(self.nIterations+1):

            # Solve Ito-SDE using Stormer-Verlet scheme
            WP_np = self.h**2 * self.compute_multivariate_normal()
            xi_n = self.xi_nm + self.h/2*np.matmul(self.C_approx, self.P_nm)

            # Gradient Log-Likelihood
            grad_LL = computeGradientFD(self.SS_X, xi_n, eps=0.000001)

            # Gradient Log-Prior
            # grad_LP = np.transpose(computeGradientFD(self.log_prior, np.transpose(xi_n)))
            grad_LP = 0  # Prior cancels for now
            grad_phi = -grad_LP - grad_LL

            P_np = self.hfm/self.hfp*self.P_nm + self.h/self.hfp * \
                (-grad_phi) + np.sqrt(self.f0)/self.hfp * \
                            np.matmul(self.inv_L_c_T, WP_np)
            xi_np = xi_n + self.h/2*np.matmul(self.C_approx, P_np)

            self.P_nm = P_np
            self.xi_nm = xi_np

            # Reverse Change variable and store values of the chain
            self.current_val = self.cv_backward(xi_n)
            self.current_fun_eval = self.f_X(self.current_val)

            # We save 100 function evaluation for the post process
            self.write_fun_eval(i, self.nIterations/100, self.current_fun_eval)

            # We estimate time after a hundred iterations
            if i == 100:
                self.compute_time(self.t1)

            # Save the next current value
            self.write_val(self.current_val)

        self.compute_covariance()

        self.terminate_loop()


def sum_of_square(data1, data2):

    J = - 1/2 * np.sum(((data1.y-data2)/data1.std_y)**2, axis=0)

    return J


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
