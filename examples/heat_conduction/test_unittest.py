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

        post_dist = pybitup.solve_problem.Sampling(input_file_name)
        post_dist.sample(heat_conduction_model)
        post_dist.__del__()

        post_dist = pybitup.solve_problem.Propagation(input_file_name)
        post_dist.propagate(heat_conduction_model)
        post_dist.__del__()

        pybitup.post_process.post_process_data(input_file_name)

        # Load solution from pce 
        f = open("output/pce_model.pickle","rb")
        model = pickle.load(f)
        f.close()

        f = open("output/pce_poly.pickle","rb")
        poly = pickle.load(f)
        f.close()

        point = np.loadtxt("output/mcmc_chain.csv",delimiter=",")
        index = []
        resp = []

        # Compute MC solution from Baeysian inference
        for i in range(len(point)):
            try:
                resp.append(np.load("output/heat_conduction_1_fun_eval."+str(i)+".npy"))
                index.append(i)
            except: pass

        resp = np.array(resp)
        respMod = model.eval(point[index])

        varMc = np.var(resp,axis=0)
        meanMc = np.mean(resp,axis=0)
        meanMod = np.mean(respMod,axis=0)
        varMod = np.var(respMod,axis=0)

        xMc = np.arange(10.0, 70.0, 4.0)
        reader = pd.read_csv('test_design_points.dat',header=None)
        xMod = reader.values[:]
        xmod_t = np.transpose(xMod)

        # Compute the two areas and compare them for the unit test 
        area_mean_Mc = np.trapz(meanMc, xMc)
        area_mean_PCE = np.trapz(meanMod, xmod_t[0])
        self.assertAlmostEqual(area_mean_Mc/100, area_mean_PCE/100, 0) 

        area_var_Mc = np.trapz(varMc, xMc)
        area_var_PCE = np.trapz(varMod, xmod_t[0])
        self.assertAlmostEqual(area_var_Mc/10, area_var_PCE/10, 0) # Equal +/- 10.0

    def test_check_2(self):
        # Check quadrature: monte-carlo with coefficients: spectral 



#unittest.main()
