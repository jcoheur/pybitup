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

                #error_bar (data_exp.x[ind_1:ind_2+1], data_exp.y[ind_1:ind_2+1], 
                        #data_exp.std_y[ind_1:ind_2+1], lineColor[i][0])

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
                numFig = 200
                if  "scatter" in inputFields["Posterior"]["estimation"]: 
                    for i in range(n_unpar):
                        range_data = range(1,len(param_value[:,i]), 10)
                        data_i = param_value[range_data, i]

                        for j in range(i+1, n_unpar):
                            data_j = param_value[range_data, j]

                            plt.figure(numFig)
                            plt.scatter(data_j, data_i, c=['C2'], s=10)
                            numFig += 1 

                saveToTikz('no_reparam.tex')


        # ----- Bivariate probability distribution functions ----- 
        # --------------------------------------------------------

        # OLD MATLAB IMPLEMENTATIL
        # figNum=1;
        # %param_values=inputParams.Parametrization(param_values);
        # if strcmp(bivariatePDF.plot, 'yes')
        #     for i = 1 : nParam_uncertain
        #         param_i = param_values(samplesPlot,i);
        #         for j = i+1 : nParam_uncertain
        #             param_j = param_values(samplesPlot,j);
        #             if strcmp(bivariatePDF.plotHist, 'yes')
        #                 figure
        #                 hist3([param_i param_j], bivariatePDF.binHist);
        #                 set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
        #                 %ylim([1.2 1.5].*1e5); xlim([0 0.5].*1e6) 
        #             end
        #             if strcmp(bivariatePDF.plotContourFilled, 'yes')
        #                 figure; hold on
        #                 [n,c] = hist3([param_i param_j], bivariatePDF.binHist);
        #                 [~,h]=contourf(c{1}, c{2}, n.', 80);
        #                 set(h,'LineColor','none')
        #             end
        #             if strcmp(bivariatePDF.plotContour, 'yes')
        #                 figure(figNum); hold on
        #                 [n,c] = hist3([param_i param_j], bivariatePDF.binHist);
        #                 [~,h]=contour(c{1}, c{2}, n.', 10);
        #                 set(h,'LineColor',matlab_default_colors(1,:))
        #             end
        #             if strcmp(bivariatePDF.scatterPlot, 'yes')
        #                 figure(figNum); hold on

        #                 scatter(param_j, param_i, 15, matlab_default_colors(2,:), 'filled')
        #             end
        #             figNum=figNum+1;
        #             xlabel(paramNames(j)); 
        #             ylabel(paramNames(i)); 
        #             run('plot_dimensions'); box off
                    
        #             if strcmp(bivariatePDF.saveImg, 'yes')
        #                 matlab2tikz(['2Dpdf_' num2str(i) '_' num2str(j) '.tex'])
        #             end
                
        #         end
        #     end
        # end



    # -------------------------------------------
    # ------ Posterior predictive check ---------
    # -------------------------------------------

    if (inputFields.get("Propagation") is not None):
        if inputFields["Propagation"]["display"] == "yes":
            plt.figure(inputFields["Propagation"]["num_plot"])

            # By default, we have saved 100 function evaluations
            n_fun_eval = 500
            delta_it = int(n_samples/n_fun_eval)

            start_val = int(inputFields["Propagation"]["burnin"]*delta_it)

            # By default, the last function evaluation to be plotted is equal to the number of iterations
            end_val = int(n_samples)

            for i in range(data_exp.n_data_set):

                data_ij_max = np.zeros(data_exp.size_x(i))
                data_ij_min = np.zeros(data_exp.size_x(i))
                ind_1 = data_exp.index_data_set[i, 0]
                ind_2 = data_exp.index_data_set[i, 1]

                # Histogram 
                data_hist = np.zeros([n_fun_eval, data_exp.size_x(i)])

                # Initialise bounds
                data_ij_max = -1e5*np.ones(data_exp.size_x(i))
                data_ij_min = 1e5*np.ones(data_exp.size_x(i))
                data_ij_mean = np.zeros(data_exp.size_x(i))
                data_ij_var = np.zeros(data_exp.size_x(i))

                for c_eval, j in enumerate(range(start_val+delta_it, end_val, delta_it)):

                    # Load current data
                    data_ij = np.load("output/fun_eval.{}.npy".format(j))
                    data_set_n = data_ij[ind_1:ind_2+1]

                    # Update bounds
                    for k in range(data_exp.size_x(i)):
                        if data_ij_max[k] < data_set_n[k]:
                            data_ij_max[k] = data_set_n[k]
                        elif data_ij_min[k] > data_set_n[k]:
                            data_ij_min[k] = data_set_n[k]

                    data_hist[c_eval, :] = data_set_n[:]  

                    # Update mean 
                    data_ij_mean[:] = data_ij_mean[:] + data_set_n[:]

                    # Plot all realisation (modify alpha value to see something)
                    #plt.plot(data_exp.x[ind_1:ind_2+1], data_set_n[:], alpha=0.)

                # Compute mean 
                data_ij_mean = data_ij_mean[:]/n_fun_eval

                # Identical loop to compute the variance 
                for j in range(start_val+delta_it, end_val, delta_it):

                    # Load current data
                    data_ij = np.load("output/fun_eval.{}.npy".format(j))
                    data_set_n = data_ij[ind_1:ind_2+1]

                    # Compute variance
                    data_ij_var = data_ij_var[:] + (data_set_n[:] - data_ij_mean[:])**2

                data_ij_var = data_ij_var[:]/(n_fun_eval - 1) 
            
                # # Plot median and all results from propagation
                # plt.plot(data_exp.x[ind_1:ind_2+1], (data_ij_min +
                #                                     data_ij_max)/2, color=lineColor[i][0], alpha=0.5)
                # plt.fill_between(data_exp.x[ind_1:ind_2+1], data_ij_min[:],
                #                 data_ij_max[:], facecolor=lineColor[i][0], alpha=0.1)

                # Plot mean and 95% confidence interval for the mean 
                # CI_lowerbound = data_ij_mean - 1.96*np.sqrt(data_ij_var/n_fun_eval)
                # CI_upperbound = data_ij_mean + 1.96*np.sqrt(data_ij_var/n_fun_eval)
                # plt.plot(data_exp.x[ind_1:ind_2+1], data_ij_mean, color=lineColor[i][0], alpha=0.5)
                # plt.fill_between(data_exp.x[ind_1:ind_2+1],  CI_lowerbound, CI_upperbound, facecolor=lineColor[i][0], alpha=0.1)

                # Plot mean 
                # ---------
                plt.plot(data_exp.x[ind_1:ind_2+1], data_ij_mean, color=lineColor[i][0], alpha=0.5)
                # Plot 95% credible interval
                # ---------------------------
                plt.fill_between(data_exp.x[ind_1:ind_2+1],  np.percentile(data_hist, 2.5, axis=0), 
                                np.percentile(data_hist, 97.5, axis=0), facecolor=lineColor[i][0], alpha=0.3)
                # Plot 95% prediction interval
                # -----------------------------
                plt.fill_between(data_exp.x[ind_1:ind_2+1],  np.percentile(data_hist, 2.5, axis=0)-.005, 
                                np.percentile(data_hist, 97.5, axis=0)+0.005, facecolor=lineColor[i][0], alpha=0.1)
                del data_ij_max, data_ij_min, data_set_n

    # Show plot   
    #saveToTikz('propagation.tex')
    plt.show()


def saveToTikz(nameTikzFile):

    plt.grid(True)
    tikz_save(nameTikzFile, figureheight='\\figureheight', figurewidth='\\figurewidth',
              extra_axis_parameters=['/pgf/number format/.cd, 1000 sep={}', 'title=\\figuretitle', 'xlabel=\\figurexlabel', 'ylabel=\\figureylabel'])
			  
			  
def error_bar (x_data, y_data, error, col, line_width=1): 

    yp_data_error = y_data + error 
    ym_data_error = y_data - error

    Dx = x_data[-1] - x_data[0]
    dx = Dx/150

    ld = len(x_data)
    for i in range(0, ld): 
        plt.plot(np.array([x_data[i], x_data[i]]), np.array([ym_data_error[i], yp_data_error[i]]), color=col)
        plt.plot(np.array([x_data[i]-dx, x_data[i]+dx]), np.array([yp_data_error[i], yp_data_error[i]]), color=col)
        plt.plot(np.array([x_data[i]-dx, x_data[i]+dx]), np.array([ym_data_error[i], ym_data_error[i]]), color=col)

