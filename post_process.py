import matplotlib.pyplot as plt
import pickle
import json
from jsmin import jsmin
import numpy as np
from scipy import stats

from matplotlib2tikz import save as tikz_save


def post_process_data(inputFields):
       
    # Colors
    lineColor = [['C0'], ['C1'], ['C2'], [
        'C3'], ['C4'], ['C5'], ['C6'], ['C7']]

    
    with open('output/output.dat', 'r') as file_param:

        for ind, line in enumerate(file_param):
            if ind == 1:
                # Get random variable names 
                c_chain = line.strip()
                unpar_name=c_chain.split()

            elif ind == 3: 
                 # Get number of iterations 
                c_chain = line.strip()
                n_iterations = int(c_chain)

        n_unpar = len(unpar_name)
        n_samples = n_iterations + 2

    # -------------------------------------------
    # --------- Plot experimental data ----------
    # -------------------------------------------

    if (inputFields.get("Data") is not None):
        # Load experimental data
        with open('output/data', 'rb') as file_data_exp:
            pickler_data_exp = pickle.Unpickler(file_data_exp)
            data_exp = pickler_data_exp.load()

        if inputFields["Data"]["display"] == "yes":
            for i in range(data_exp.n_data_set):

                ind_1 = data_exp.index_data_set[i, 0]
                ind_2 = data_exp.index_data_set[i, 1]

                plt.figure(inputFields["Data"]["num_plot"])
                plt.plot(data_exp.x[ind_1:ind_2+1], data_exp.y[ind_1:ind_2+1],
                        'o', color=lineColor[i][0], mfc='none')
                #, edgecolors='r'

    # -------------------------------------------
    # --------- Plot initial guess --------------
    # -------------------------------------------

    if (inputFields.get("InitialGuess") is not None):
        if inputFields["InitialGuess"]["display"] == "yes":
            data_init = np.load("output/fun_eval.{}.npy".format(0))

            for i in range(data_exp.n_data_set):

                ind_1 = data_exp.index_data_set[i, 0]
                ind_2 = data_exp.index_data_set[i, 1]

                plt.figure(inputFields["InitialGuess"]["num_plot"])
                plt.plot(data_exp.x[ind_1:ind_2+1],
                        data_init[ind_1:ind_2+1], '--', color=lineColor[i][0])




    if (inputFields.get("MarkovChain") is not None) or (inputFields.get("Posterior") is not None) or  (inputFields.get("Propagation") is not None):

        # Load the samples of the distribution                        
        param_value_raw = np.zeros((n_iterations+2, n_unpar))
        with open('output/mcmc_chain2.dat', 'r') as file_param:
            i = 0
            for line in file_param:
                c_chain = line.strip()
                param_value_raw[i, :] = np.fromstring(
                    c_chain[1:len(c_chain)-1], sep=' ')
                i += 1

        # -------------------------------------------
        # --------- Plot markov chains --------------
        # -------------------------------------------

        if inputFields.get("MarkovChain") is not None and inputFields["MarkovChain"]["display"] == "yes":

            for i in range(n_unpar):
                plt.figure(100+i)
                plt.plot(range(n_samples), param_value_raw[:, i])
                plt.xlabel("Number of iterations")
                plt.ylabel(unpar_name[i])



        # -------------------------------------------
        # -------- Posterior distribution -----------
        # -------------------------------------------
    
        if inputFields.get("Posterior") is not None and inputFields["Posterior"]["display"] == "yes":

            burnin_it = inputFields["Posterior"]["burnin"]
            param_value = param_value_raw[range(burnin_it, n_samples), :]

            if inputFields["Posterior"]["distribution"] == "marginal":

                if "ksdensity" in inputFields["Posterior"]["estimation"]:
                    for i in range(n_unpar):

                        # Estimate marginal pdf using gaussian kde
                        data_i = param_value[:, i]
                        kde = stats.gaussian_kde(data_i)
                        x = np.linspace(data_i.min(), data_i.max(), 100)
                        p = kde(x)

                        # Plot 
                        plt.figure(200+i)
                        plt.plot(x, p)
                        plt.xlabel(unpar_name[i])
                        plt.ylabel("Probability density")

                        # Find and plot the mode
                        plt.plot(x[np.argmax(p)], p.max(), 'r*')

                        if inputFields["Posterior"]["distribution"] == "yes":
                            saveToTikz("marginal_pdf_param_"+i+".tex")

                if "hist" in inputFields["Posterior"]["estimation"]:
                     for i in range(n_unpar):

                        data_i = param_value[:, i]

                        plt.figure(200+i)
                        plt.hist(data_i, bins='auto', density=True)
               

            if inputFields["Posterior"]["distribution"] == "bivariate":
                # Compute bivariate marginal pdf 
                a = 1



    # -------------------------------------------
    # ------ Posterior predictive check ---------
    # -------------------------------------------

    if (inputFields.get("Propagation") is not None):
        if inputFields["Propagation"]["display"] == "yes":
            plt.figure(inputFields["Propagation"]["num_plot"])

            # By default, we have saved 100 function evaluations
            delta_it = int(n_samples/100)

            start_val = int(inputFields["Propagation"]["burnin"]*delta_it)

            # By default, the last function evaluation to be plotted is equal to the number of iterations
            end_val = int(n_samples)

            for i in range(data_exp.n_data_set):

                data_ij_max = np.zeros(data_exp.size_x(i))
                data_ij_min = np.zeros(data_exp.size_x(i))
                ind_1 = data_exp.index_data_set[i, 0]
                ind_2 = data_exp.index_data_set[i, 1]

                # Initialise bounds
                data_i1 = np.load("output/fun_eval.{}.npy".format(start_val))
                data_ij_max = -1e5*np.ones(data_exp.size_x(i))
                data_ij_min = 1e5*np.ones(data_exp.size_x(i))

                for j in range(start_val+delta_it, end_val+1, delta_it):

                    # Load current data
                    data_ij = np.load("output/fun_eval.{}.npy".format(j))
                    data_set_n = data_ij[ind_1:ind_2+1]

                    # Update bounds
                    for k in range(data_exp.size_x(i)):
                        if data_ij_max[k] < data_set_n[k]:
                            data_ij_max[k] = data_set_n[k]
                        elif data_ij_min[k] > data_set_n[k]:
                            data_ij_min[k] = data_set_n[k]

                    plt.plot(data_exp.x[ind_1:ind_2+1], data_set_n[:], alpha=0.0)

                plt.fill_between(data_exp.x[ind_1:ind_2+1], data_ij_min[:],
                                data_ij_max[:], facecolor=lineColor[i][0], alpha=0.1)
                plt.plot(data_exp.x[ind_1:ind_2+1], (data_ij_min +
                                                    data_ij_max)/2, color=lineColor[i][0], alpha=0.5)

                del data_ij_max, data_ij_min, data_set_n

    # Show plot
    # saveToTikz('test.tex')
    plt.show()


def saveToTikz(nameTikzFile):

    plt.grid(True)
    tikz_save(nameTikzFile, figureheight='\\figureheight', figurewidth='\\figurewidth',
              extra_axis_parameters=['/pgf/number format/.cd, 1000 sep={}', 'title=\\figuretitle', 'xlabel=\\figurexlabel', 'ylabel=\\figureylabel'])
