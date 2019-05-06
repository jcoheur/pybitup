import matplotlib.pyplot as plt
import pickle
import json
from jsmin import jsmin
import numpy as np

from matplotlib2tikz import save as tikz_save


def post_process_data(input_file_name):

    # Open and read input file
    # -------------------------
    # First, remove comments from the file with jsmin because json doesn't allow it
    with open("{}".format(input_file_name)) as js_file:
        minified = jsmin(js_file.read())
    user_inputs = json.loads(minified)
    # Previous solution: if there is no comment in the file
    #with open("heat_capacity.json", 'r') as input_file:
    #   user_inputs = json.load(input_file)

    # Load experimental data
    with open('output/data', 'rb') as file_data_exp:
        pickler_data_exp = pickle.Unpickler(file_data_exp)
        data_exp = pickler_data_exp.load()

    # Colors
    lineColor = [['C0'], ['C1'], ['C2'], [
        'C3'], ['C4'], ['C5'], ['C6'], ['C7']]

    # -------------------------------------------
    # --------- Plot experimental data ----------
    # -------------------------------------------

    if (user_inputs["PostProcess"].get("Data") is not None):
        if user_inputs["PostProcess"]["Data"]["display"] == "yes":
            for i in range(data_exp.n_data_set):

                ind_1 = data_exp.index_data_set[i, 0]
                ind_2 = data_exp.index_data_set[i, 1]

                plt.figure(user_inputs["PostProcess"]["Data"]["num_plot"])
                plt.plot(data_exp.x[ind_1:ind_2+1], data_exp.y[ind_1:ind_2+1],
                        'o', color=lineColor[i][0], mfc='none')
                #, edgecolors='r'

    # -------------------------------------------
    # --------- Plot initial guess --------------
    # -------------------------------------------

    if (user_inputs["PostProcess"].get("InitialGuess") is not None):
        if user_inputs["PostProcess"]["InitialGuess"]["display"] == "yes":
            data_init = np.load("output/fun_eval.{}.npy".format(0))

            for i in range(data_exp.n_data_set):

                ind_1 = data_exp.index_data_set[i, 0]
                ind_2 = data_exp.index_data_set[i, 1]

                plt.figure(user_inputs["PostProcess"]["InitialGuess"]["num_plot"])
                plt.plot(data_exp.x[ind_1:ind_2+1],
                        data_init[ind_1:ind_2+1], '--', color=lineColor[i][0])

    # -------------------------------------------
    # --------- Plot markov chains --------------
    # -------------------------------------------

    if (user_inputs["PostProcess"].get("MarkovChain") is not None):
        if user_inputs["PostProcess"]["MarkovChain"]["display"] == "yes" and user_inputs['Inference']['algorithm'] != "None":
            n_iterations = int(user_inputs['Inference']
                            ['algorithm']['n_iterations'])
            n_unpar = len(user_inputs['Inference']['param'])
            param_value = np.zeros((n_iterations+2, n_unpar))
            with open('output/mcmc_chain.dat', 'r') as file_param:
                i = 0
                for line in file_param:
                    c_chain = line.strip()
                    param_value[i, :] = np.fromstring(
                        c_chain[1:len(c_chain)-1], sep=' ')
                    i += 1

            for i in range(n_unpar):
                plt.figure(100+i)
                plt.plot(range(n_iterations+2), param_value[:, i])



    # -------------------------------------------
    # -------- Posterior distribution -----------
    # -------------------------------------------
    
    if (user_inputs["PostProcess"].get("Posterior") is not None):
        if user_inputs["PostProcess"]["Posterior"]["display"] == "yes":
            if user_inputs["PostProcess"]["Posterior"]["distribution"] == "marginal":
                n_unpar = len(user_inputs['Inference']['param'])
                param_value = np.zeros((100, n_unpar))
                for i in range(n_unpar):

                    #[F, XI] = ksdensity(param_i)
                    #xlabel(param_name{:})
                    #plt.plot(XI, F)
                    #indp = find(XI >= meanp(i), 1, 'first')
                    #plt.plot(XI(indp), F(indp), 'r*')

                    if user_inputs["PostProcess"]["Posterior"]["distribution"] == "yes":
                        saveToTikz("marginal_pdf_param_"+i+".tex")

            if user_inputs["PostProcess"]["Posterior"]["estimation"] == "ksdensity":
                a = 1



    # -------------------------------------------
    # ------ Posterior predictive check ---------
    # -------------------------------------------

    if (user_inputs["PostProcess"].get("Propagation") is not None):
        if user_inputs["PostProcess"]["Propagation"]["display"] == "yes":
            plt.figure(user_inputs["PostProcess"]["Propagation"]["num_plot"])

            # By default, we have saved 100 function evaluations
            delta_it = int(user_inputs["Inference"]
                        ["algorithm"]["n_iterations"]/100)

            start_val = int(user_inputs["PostProcess"]
                            ["Propagation"]["burnin"]*delta_it)

            # By default, the last function evaluation to be plotted is equal to the number of iterations
            end_val = int(user_inputs["Inference"]["algorithm"]["n_iterations"])

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
    print(plt.figure(user_inputs["PostProcess"]["Propagation"]["num_plot"]))
    
    # saveToTikz('test.tex')
    plt.show()


def saveToTikz(nameTikzFile):

    plt.grid(True)
    tikz_save(nameTikzFile, figureheight='\\figureheight', figurewidth='\\figurewidth',
              extra_axis_parameters=['/pgf/number format/.cd, 1000 sep={}', 'title=\\figuretitle', 'xlabel=\\figurexlabel', 'ylabel=\\figureylabel'])
