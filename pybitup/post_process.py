import matplotlib.pyplot as plt
import pickle
import json
from jsmin import jsmin
import numpy as np
from scipy import stats

from tikzplotlib import save as tikz_save


def post_process_data(input_file_name):
       
    # Colors
    lineColor = [['C0'], ['C1'], ['C2'], [
        'C3'], ['C4'], ['C5'], ['C6'], ['C7']]

    # -------------------------
    # Open and read input file 
    # -------------------------

    # First, remove comments from the file with jsmin because json doesn't allow it
    with open("{}".format(input_file_name)) as js_file:
        minified = jsmin(js_file.read())
    user_inputs = json.loads(minified)

    if (user_inputs.get("PostProcess") is not None):
        inputFields = user_inputs["PostProcess"] 
    else: 
        raise ValueError('Ask for post processing data but no inputs were provided')

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
        n_samples = n_iterations + 1

        unpar_name_list = {}
        for i, name_param in enumerate(unpar_name):
            unpar_name_list[name_param] = i

        num_fig = 0

    # -------------------------------------------
    # --------- Plot experimental data ----------
    # -------------------------------------------

    # if (inputFields.get("Data") is not None):
    #     # Load experimental data
    #     with open('output/data', 'rb') as file_data_exp:
    #         pickler_data_exp = pickle.Unpickler(file_data_exp)
    #         data_exp = pickler_data_exp.load()

    #     if inputFields["Data"]["display"] == "yes":
    #         for i in range(data_exp.n_data_set):

    #             ind_1 = data_exp.index_data_set[i, 0]
    #             ind_2 = data_exp.index_data_set[i, 1]

    #             plt.figure(inputFields["Data"]["num_plot"])
    #             plt.plot(data_exp.x[ind_1:ind_2+1], data_exp.y[ind_1:ind_2+1],
    #                     'o', color=lineColor[i][0], mfc='none')

    #             error_bar (data_exp.x[ind_1:ind_2+1], data_exp.y[ind_1:ind_2+1], 
    #                     data_exp.std_y[ind_1:ind_2+1], lineColor[i][0])

    #             #, edgecolors='r'

    if (inputFields.get("Data") is not None):
        # Load experimental data
        with open('output/data', 'rb') as file_data_exp:
            pickler_data_exp = pickle.Unpickler(file_data_exp)
            data_exp = pickler_data_exp.load()

        if inputFields["Data"]["display"] == "yes":
            for num_data_set, data_id in enumerate(data_exp.keys()):

                n_x = len(data_exp[data_id].x)
                n_data_set = int(len(data_exp[data_id].y)/n_x)
 
                for i in range(n_data_set): 

                    plt.figure(num_fig+i)
                    plt.plot(data_exp[data_id].x, data_exp[data_id].y[i*n_x:(i+1)*n_x],
                            'o', color=lineColor[num_data_set][0], mfc='none')

                    error_bar (data_exp[data_id].x, data_exp[data_id].y[i*n_x:i*n_x+n_x], 
                            data_exp[data_id].std_y[i*n_x:i*n_x+n_x], lineColor[num_data_set][0])

                #, edgecolors='r'


    # -------------------------------------------
    # --------- Plot initial guess --------------
    # -------------------------------------------

    if (inputFields.get("InitialGuess") is not None):
        if inputFields["InitialGuess"]["display"] == "yes":

            for num_data_set, data_id in enumerate(data_exp.keys()):
                data_init = np.load("output/{}_fun_eval.{}.npy".format(data_id, 0))

                n_x = len(data_exp[data_id].x)
                n_data_set = int(len(data_exp[data_id].y)/n_x)


                for i in range(n_data_set): 

                    plt.figure(num_fig+i)

                    plt.plot(data_exp[data_id].x,
                            data_init[i*n_x:(i+1)*n_x], '--', color=lineColor[num_data_set][0])

    if (inputFields.get("MarkovChain") is not None) or (inputFields.get("Posterior") is not None) or  (inputFields.get("Propagation") is not None):

        # Load the samples of the distribution                        
        param_value_raw = np.zeros((n_samples, n_unpar))
        with open('output/mcmc_chain.dat', 'r') as file_param:
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
            num_fig = 100
            for i in range(n_unpar):
                plt.figure(num_fig+i)
                plt.plot(range(n_samples), param_value_raw[:, i])
                plt.xlabel("Number of iterations")
                plt.ylabel(unpar_name[i])

                saveToTikz('markov_chain_'+unpar_name[i]+'.tex')

                c_mean_val = np.mean(param_value_raw[:, i])
                c_std_val = np.std(param_value_raw[:, i])
                print("{}. Mean value: {}; standard dev.: {}.".format(unpar_name[i], c_mean_val, c_std_val))





        # -------------------------------------------
        # -------- Posterior distribution -----------
        # -------------------------------------------
    
        if inputFields.get("Posterior") is not None and inputFields["Posterior"]["display"] == "yes":

            burnin_it = inputFields["Posterior"]["burnin"]
            param_value = param_value_raw[range(burnin_it, n_samples), :]

            num_fig = 200

            if inputFields["Posterior"]["distribution"] == "marginal":

                if "ksdensity" in inputFields["Posterior"]["estimation"]:
                    for i in range(n_unpar):

                        # Estimate marginal pdf using gaussian kde
                        data_i = param_value[:, i]
                        kde = stats.gaussian_kde(data_i)
                        x = np.linspace(data_i.min(), data_i.max(), 100)
                        p = kde(x)

                        # Plot 
                        plt.figure(num_fig+i)
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

                        plt.figure(num_fig+i)
                        plt.hist(data_i, bins='auto', density=True)
               
                for i in range(n_unpar): 
                    plt.figure(num_fig+i)
                    saveToTikz('marginal_pdf_'+inputFields["Posterior"]["estimation"]+'_'+unpar_name[i]+'.tex')

            if inputFields["Posterior"]["distribution"] == "bivariate":
                # Compute bivariate marginal pdf 

                if  "scatter" in inputFields["Posterior"]["estimation"]: 
                    for i in range(n_unpar):
                        range_data = range(1,len(param_value[:,i]), 10)
                        data_i = param_value[range_data, i]

                        for j in range(i+1, n_unpar):
                            data_j = param_value[range_data, j]

                            plt.figure(num_fig)
                            plt.scatter(data_j, data_i, c=['C2'], s=10)
                            num_fig += 1 

                if "contour" in inputFields ["Posterior"]["estimation"]:

                    for i, var_name in enumerate(unpar_name):

                        # Get first coordinate param values
                        x = param_value[:, i]
                        xmin = np.min(x) 
                        xmax = np.max(x) 

                        # Get second coordinatee param values
                        for var_name_2 in unpar_name[i+1:len(unpar_name)]:
                            # Number of the corresponding parameter name 
                            k = unpar_name_list[var_name_2]

                            y = param_value[:, k]
                            ymax = np.max(y)
                            ymin = np.min(y)

                            # Peform the kernel density estimate
                            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
           
                            positions = np.vstack([xx.ravel(), yy.ravel()])
                            values = np.vstack([x, y])

                            kernel = stats.gaussian_kde(values)
                            f = np.reshape(kernel(positions).T, xx.shape)

                            fig = plt.figure(num_fig)
                            ax = fig.gca()

                            ax.set_xlim(xmin, xmax)
                            ax.set_ylim(ymin, ymax)
                            # Contourf plot
                            cfset = ax.contourf(xx, yy, f, cmap='Blues')
                            ## Or kernel density estimate plot instead of the contourf plot
                            #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
                            # Contour plot
                            cset = ax.contour(xx, yy, f, colors='k')
                            # Label plot
                            ax.clabel(cset, inline=1, fontsize=10)
                            plt.xlabel(var_name)
                            plt.ylabel(var_name_2)

                            num_fig = num_fig + 1

                            saveToTikz('bivariate_contour_'+var_name+'_'+var_name_2+'.tex')

                            # plt.figure(num_fig)
                            # plt.scatter(np.log(x), np.log(y), c=['C2'], s=10)
                            # plt.xlabel(var_name)
                            # plt.ylabel(var_name_2)

                            # num_fig = num_fig + 1
                        

                #saveToTikz('no_reparam.tex')


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

            # By default, we have saved 100 function evaluations
            n_fun_eval = 100
            delta_it = int(n_samples/n_fun_eval)

            start_val = int(inputFields["Propagation"]["burnin"]*delta_it)

            # By default, the last function evaluation to be plotted is equal to the number of iterations
            end_val = int(n_samples)

            #for i in range(data_exp.n_data_set):
            for num_data_set, data_id in enumerate(data_exp.keys()):
                n_x = len(data_exp[data_id].x)
                n_data_set = int(len(data_exp[data_id].y)/n_x)
 
                for i in range(n_data_set): 
                    plt.figure(i)

                    # Initialise bounds
                    data_ij_max = -1e5*np.ones(n_x)
                    data_ij_min = 1e5*np.ones(n_x)
                    data_ij_mean = np.zeros(n_x)
                    data_ij_var = np.zeros(n_x)

                    ind_1 = i*n_x
                    ind_2 =(i+1)*n_x

                    # Histogram 
                    data_hist = np.zeros([n_fun_eval, n_x])

                    for c_eval, j in enumerate(range(start_val+delta_it, end_val, delta_it)):

                        # Load current data
                        data_ij = np.load("output/{}_fun_eval.{}.npy".format(data_id, j))
                        data_set_n = data_ij[ind_1:ind_2]

                        # Update bounds
                        for k in range(n_x):
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
                        data_ij = np.load("output/{}_fun_eval.{}.npy".format(data_id, j))
                        data_set_n = data_ij[ind_1:ind_2]

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
                    plt.plot(data_exp[data_id].x, data_ij_mean, color=lineColor[num_data_set][0], alpha=0.5)
                    # Plot 95% credible interval
                    # ---------------------------
                    plt.fill_between(data_exp[data_id].x,  np.percentile(data_hist, 2.5, axis=0), 
                                    np.percentile(data_hist, 97.5, axis=0), facecolor=lineColor[num_data_set][0], alpha=0.3)
                    # Plot 95% prediction interval
                    # -----------------------------
                    # For the prediction interval, we add the std to the result

                    plt.fill_between(data_exp[data_id].x,  np.percentile(data_hist, 2.5, axis=0)-data_exp[data_id].std_y[ind_1:ind_2], 
                                    np.percentile(data_hist, 97.5, axis=0)+data_exp[data_id].std_y[ind_1:ind_2], facecolor=lineColor[num_data_set][0], alpha=0.1)
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

