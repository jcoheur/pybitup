# Test unit test 

import unittest 

import sys
sys.path.append('../../')
import shutil

import heat_conduction
import pybitup

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle

class TestPCE(unittest.TestCase): 

    def test_pce_only(self): 
        """ Test pce using implemented distributions for input pdfs. """

        case_name = "heat_conduction_pce_only"

        # Compute solution 
        self.compute_pce(case_name)

        # Load default solution and solution from pce 
        self.load_solution(case_name, 'mean.npy', 'var.npy')

        # Compute the two areas and compare them for the unit test 
        self.compute_areas() 

        # Uncomment if needed: check the PCE and MC solutions visually 
        # self.plot_figure() 

        # Assert 
        self.assertAlmostEqual(self.area_mean_MC/100, self.area_mean_PCE/100, 0) # Equal +/- 10.0
        self.assertAlmostEqual(self.area_var_MC/100, self.area_var_PCE/100, 0) # Equal +/- 10.0

        # Clean output 
        shutil.rmtree('output')

    def test_check_2(self):
        """ Check quadrature: fekete with coefficients: lars """

        case_name = "heat_conduction_2"

        # Compute solution 
        self.compute_pce(case_name)

        # Load default solution and solution from pce 
        self.load_solution(case_name, 'mean_from_mcmc.npy', 'var_from_mcmc.npy')

        # Compute the two areas and compare them for the unit test 
        self.compute_areas() 

        # Uncomment if needed: check the PCE and MC solutions visually 
        # self.plot_figure() 

        # Assert 
        self.assertAlmostEqual(self.area_mean_MC/100, self.area_mean_PCE/100, 0) 
        self.assertAlmostEqual(self.area_var_MC/10, self.area_var_PCE/10, 0) # Equal +/- 10.0

        # Clean output 
        shutil.rmtree('output')


    # def test_check_3(self):
    #     """ Check quadrature: simplex with coefficients: lasso
    #     DOESNT WORK: method='revised_simplex' is not in linprog, and 'method'=simplex doesn't produce correct results. """

    #     case_name = "heat_conduction_3"

    #     # Compute solution 
    #     self.compute_solution(case_name)

    #     # Load default solution and solution from pce 
    #     self.load_solution(case_name, 'mean_from_mcmc.npy', 'var_from_mcmc.npy')

    #     # Compute the two areas and compare them for the unit test 
    #     self.compute_areas() 

    #     # Assert 
    #     self.assertAlmostEqual(self.area_mean_MC/100, self.area_mean_PCE/100, 0) 
    #     self.assertAlmostEqual(self.area_var_MC/10, self.area_var_PCE/10, 0) # Equal +/- 10.0

    #     # Clean output 
    #     shutil.rmtree('output')

    def test_check_4(self):
        """ Check quadrature: positive newton with coefficients: sepctral """

        case_name = "heat_conduction_4"

        # Compute solution 
        self.compute_pce(case_name)
       
        # Load default solution and solution from pce 
        self.load_solution(case_name, 'mean_from_mcmc.npy', 'var_from_mcmc.npy')

        # Compute the two areas and compare them for the unit test 
        self.compute_areas() 

        # Uncomment if needed: check the PCE and MC solutions visually 
        # self.plot_figure() 

        # Assert 
        self.assertAlmostEqual(self.area_mean_MC/100, self.area_mean_PCE/100, 0) 
        self.assertAlmostEqual(self.area_var_MC/10, self.area_var_PCE/10, 0)

        # Clean output 
        shutil.rmtree('output')

    def test_bayes_and_pce(self):
        """ Test quadrature: monte-carlo with coefficients: spectral 
        Test that pce can be run using samples from bayesian inference. """

        # Run 
        case_name = "heat_conduction_1"
        input_file_name = "{}.json".format(case_name) 

        # Sample from Bayesian formula
        heat_conduction_model = {}
        heat_conduction_model[case_name] = heat_conduction.HeatConduction()
        post_dist = pybitup.solve_problem.Sampling(input_file_name)
        post_dist.sample(heat_conduction_model)
        post_dist.__del__()

        # Compute PCE 
        self.compute_pce(case_name)

        # Load solution from pce (self.meanMC and self.varMC computed later)
        self.load_solution(case_name, 'mean_from_mcmc.npy', 'var_from_mcmc.npy')

        # Compute MC solution from Bayesian inference
        point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
        index = []
        resp = []
        for i in range(len(point)):
            try:
                resp.append(np.load("output/heat_conduction_1_fun_eval."+str(i)+".npy"))
                index.append(i)
            except: pass
        resp = np.array(resp)
        self.meanMC = np.mean(resp,axis=0)
        self.varMC = np.var(resp,axis=0)

        # Compute the two areas and compare them for the unit test 
        self.compute_areas() 

        # Uncomment if needed: check the PCE and MC solutions visually 
        # self.plot_figure() 

        # Assert 
        self.assertAlmostEqual(self.area_mean_MC/100, self.area_mean_PCE/100, 0) 
        self.assertAlmostEqual(self.area_var_MC/10, self.area_var_PCE/10, 0)
        
        # Clean output 
        shutil.rmtree('output')

    def compute_pce(self, case_name): 

            input_file_name = "{}.json".format(case_name) 

            heat_conduction_model = {}
            heat_conduction_model[case_name] = heat_conduction.HeatConduction()

            post_dist = pybitup.solve_problem.Propagation(input_file_name)
            post_dist.propagate(heat_conduction_model)
            post_dist.__del__()


    def load_solution(self, case_name, mean_file_name, var_file_name): 
        # File related to pce is located in output by default with pce_model and pce_poly 

        f = open("output/propagation/pce_model_"+case_name+".pickle","rb")
        model = pickle.load(f)
        f.close()

        f = open("output/propagation/pce_poly_"+case_name+".pickle","rb")
        poly = pickle.load(f)
        f.close()

        reader = pd.read_csv('test_design_points.dat',header=None)
        xMod = reader.values[:]

        self.xMC = np.arange(10.0, 70.0, 4.0)
        self.meanMC = np.load(mean_file_name)
        self.varMC = np.load(var_file_name)
        self.xmod_t = np.transpose(xMod)
        self.meanMod = model.mean
        self.varMod = model.var
    
    def compute_areas(self): 

        self.area_mean_MC = np.trapz(self.meanMC, self.xMC)
        self.area_mean_PCE = np.trapz(self.meanMod, self.xmod_t[0])
        self.area_var_MC = np.trapz(self.varMC, self.xMC)
        self.area_var_PCE = np.trapz(self.varMod, self.xmod_t[0])

    def plot_figure(self): 
        """ Use to check visually the solution from pce and MC."""

        plt.figure(1)
        plt.rcParams.update({"font.size":16})
        plt.plot(self.xmod_t[0], self.meanMod,'C0',label="PCE")
        plt.plot(self.xMC,self.meanMC,'C1--',label="MC")
        plt.legend(prop={'size':16})
        plt.ylabel("Mean")
        plt.xlabel("x")
        plt.grid()

        plt.figure(2)
        plt.rcParams.update({"font.size":16})
        plt.plot(self.xmod_t[0],self.varMod,'C0',label="PCE")
        plt.plot(self.xMC,self.varMC,'C1--',label="MC")
        plt.legend(prop={'size':16})
        plt.ylabel("Variance")
        plt.xlabel("x")
        plt.grid()

        # plt.figure(3)
        # plt.rcParams.update({"font.size":16})
        # plt.plot(point[:,0],point[:,1],".C0")
        # plt.xlabel("$O$")
        # plt.ylabel("$h$")
        # plt.grid()


        plt.show()
