# Test unit test 

import unittest 

import sys
sys.path.append('../../')

import heat_conduction
import pybitup

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle

class TestPCE(unittest.TestCase): 

    def test_pce_only(self): 

        # Run 
        case_name = "heat_conduction_pce_only"
        input_file_name = "{}.json".format(case_name) 

        heat_conduction_model = {}
        heat_conduction_model[case_name] = heat_conduction.HeatConduction()

        post_dist = pybitup.solve_problem.Propagation(input_file_name)
        post_dist.propagate(heat_conduction_model)
        post_dist.__del__()

        # Load solution from pce 
        f = open("output/pce_model.pickle","rb")
        model = pickle.load(f)
        f.close()

        f = open("output/pce_poly.pickle","rb")
        poly = pickle.load(f)
        f.close()

        reader = pd.read_csv('test_design_points.dat',header=None)
        xMod = reader.values[:]

        xMc = np.arange(10.0, 70.0, 4.0)
        varMc = np.load('var.npy')
        meanMc = np.load('mean.npy')
        meanMod = model.mean
        varMod = model.var

        # Compute the two areas and compare them for the unit test 
        area_mean_Mc = np.trapz(meanMc, xMc)
        xmod_t = np.transpose(xMod)
        area_mean_PCE = np.trapz(meanMod, xmod_t[0])
        self.assertAlmostEqual(area_mean_Mc/100, area_mean_PCE/100, 0) # Equal +/- 10.0

        area_var_Mc = np.trapz(varMc, xMc)
        area_var_PCE = np.trapz(varMod, xmod_t[0])
        self.assertAlmostEqual(area_var_Mc/100, area_var_PCE/100, 0) # Equal +/- 10.0

    def test_check_1(self):
        # Check quadrature: monte-carlo with coefficients: spectral 

        # Run 
        case_name = "heat_conduction_1"
        input_file_name = "{}.json".format(case_name) 

        heat_conduction_model = {}
        heat_conduction_model[case_name] = heat_conduction.HeatConduction()

        post_dist = pybitup.solve_problem.Propagation(input_file_name)
        post_dist.propagate(heat_conduction_model)
        post_dist.__del__()

        # Load solution from pce 
        f = open("output/pce_model.pickle","rb")
        model = pickle.load(f)
        f.close()

        f = open("output/pce_poly.pickle","rb")
        poly = pickle.load(f)
        f.close()

        xMC = np.arange(10.0, 70.0, 4.0)
        varMC = np.load('var_from_mcmc.npy')
        meanMC = np.load('mean_from_mcmc.npy')
        meanMod = model.mean
        varMod = model.var
        
        reader = pd.read_csv('test_design_points.dat',header=None)
        xMod = reader.values[:]
        xmod_t = np.transpose(xMod)

        plt.figure(1)
        plt.rcParams.update({"font.size":16})
        plt.plot(xMod,meanMod,'C0',label="PCE")
        plt.plot(xMC,meanMC,'C1--',label="MC")
        plt.legend(prop={'size':16})
        plt.ylabel("Mean")
        plt.xlabel("x")
        plt.grid()

        plt.figure(2)
        plt.rcParams.update({"font.size":16})
        plt.plot(xMod,varMod,'C0',label="PCE")
        plt.plot(xMC,varMC,'C1--',label="MC")
        plt.legend(prop={'size':16})
        plt.ylabel("Variance")
        plt.xlabel("x")
        plt.grid()


        plt.show()

        # Compute the two areas and compare them for the unit test 
        area_mean_MC = np.trapz(meanMC, xMC)
        area_mean_PCE = np.trapz(meanMod, xmod_t[0])
        self.assertAlmostEqual(area_mean_MC/100, area_mean_PCE/100, 0) 

        area_var_MC = np.trapz(varMC, xMC)
        area_var_PCE = np.trapz(varMod, xmod_t[0])
        self.assertAlmostEqual(area_var_MC/10, area_var_PCE/10, 0) # Equal +/- 10.0

    def test_check_2(self):
        # Check quadrature: monte-carlo with coefficients: spectral 

        case_name = "heat_conduction_2"
        input_file_name = "{}.json".format(case_name) 

        heat_conduction_model = {}
        heat_conduction_model[case_name] = heat_conduction.HeatConduction()

        post_dist = pybitup.solve_problem.Propagation(input_file_name)
        post_dist.propagate(heat_conduction_model)
        post_dist.__del__()

        f = open("output/pce_model.pickle","rb")
        model = pickle.load(f)
        f.close()

        f = open("output/pce_poly.pickle","rb")
        poly = pickle.load(f)
        f.close()

        # %% Monte Carlo and error
        xMC = np.arange(10.0, 70.0, 4.0)
        varMC = np.load('var_from_mcmc.npy')
        meanMC = np.load('mean_from_mcmc.npy')
        meanMod = model.mean
        varMod = model.var
        
        reader = pd.read_csv('test_design_points.dat',header=None)
        xMod = reader.values[:]
        xmod_t = np.transpose(xMod)

        # %% Figures

        plt.figure(1)
        plt.rcParams.update({"font.size":16})
        plt.plot(xMod,meanMod,'C0',label="PCE")
        plt.plot(xMC,meanMC,'C1--',label="MC")
        plt.legend(prop={'size':16})
        plt.ylabel("Mean")
        plt.xlabel("x")
        plt.grid()

        plt.figure(2)
        plt.rcParams.update({"font.size":16})
        plt.plot(xMod,varMod,'C0',label="PCE")
        plt.plot(xMC,varMC,'C1--',label="MC")
        plt.legend(prop={'size':16})
        plt.ylabel("Variance")
        plt.xlabel("x")
        plt.grid()

        plt.show()

        # Compute the two areas and compare them for the unit test 
        area_mean_MC = np.trapz(meanMC, xMC)
        area_mean_PCE = np.trapz(meanMod, xmod_t[0])
        self.assertAlmostEqual(area_mean_MC/100, area_mean_PCE/100, 0) 

        area_var_MC = np.trapz(varMC, xMC)
        area_var_PCE = np.trapz(varMod, xmod_t[0])
        self.assertAlmostEqual(area_var_MC/10, area_var_PCE/10, 0) # Equal +/- 10.0



#unittest.main()
