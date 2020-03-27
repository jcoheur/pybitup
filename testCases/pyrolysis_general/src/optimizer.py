import matplotlib.pyplot as plt
import spotpy
import pandas as pd
import re
import src.pyrolysis as pyro
import matplotlib.cm as cm
import numpy as np
import logging
import time
import os
import shutil
from src.read_experiments import ReadExperiments

def rmse_multiple_files(evaluation, simulation):
    """
    Root Mean Squared Error for more than 1 file

        .. math::

         RMSE=\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}

    :evaluation: Observed data to compared with simulation data.
    :type: list of lists

    :simulation: simulation data to compared with evaluation data
    :type: list of lists

    :return: Root Mean Squared Error
    :rtype: float
    """

    scale_coeff_dRho = 10000
    mses = []
    n_objectives = len(evaluation)  # 1 if 1 objective (dRho), 2 if 2 objectives (dRho, Rho)
    # n_objectives = 1  # 1 if only dRho
    n_files = len(evaluation[0])  # number of files to optimize
    for i in range(n_objectives):
        for j in range(n_files):
            mse_calc = spotpy.objectivefunctions.mse(evaluation[i][j], simulation[i][j])
            if i == 0:  # this means we are in dRho
                mses.append(mse_calc*scale_coeff_dRho)
            else:
                mses.append(mse_calc)
    return np.sqrt(sum(mses))


def get_numbers_from_filename(filename):
    """
    This is used to get the heating rate from the filename

    :param filename: str (with the heating rate)
    :return: str
    """
    return re.findall(r"[-+]?\d*\.\d+|\d+", filename)


class spotpy_setup(object):
    """Class to setup spotpy optimization routine

    Attributes:
        :atrr betas:
        :atrr params:
        :atrr names:
        :atrr times:
        :atrr dRho:
        :atrr Rho:
        :atrr temperatures:
        :atrr iternumber:
    """

    def __init__(self, files, params, folder, scheme_file, pyro_type, keepFolders=False, isothermal=False):
        """

        :param files: list of files to be treated
        :param params: list of params to be optimized
        :param folder: str folder where to read the experiments
        :param scheme_file: str with the scheme used
        :param pyro_type: str for type of pyrolysis (parallel, competitive, etc)
        :param keepFolders: bool if folder of simulations are saved
        :param isothermal: bool isothermal tests (not used in most cases)
        """
        self.pyro_type = pyro_type
        self.folder = folder
        self.files = files
        self.betas = []
        for filename in self.files:
            self.betas.append(float(get_numbers_from_filename(filename)[0]))
        self.params = params
        self.names = []
        self.get_param_names(params)
        self.times = []
        self.dRho  = []
        self.Rho   = []
        self.temperatures = []
        self.scheme_file = scheme_file
        for filename in self.files:
            self.read_experiments(filename, folder)

        self.iternumber = 0
        self.keepFolders = keepFolders

        # X = self.dRho+self.Rho

    def get_param_names(self,params):
        """

        :param params:
        """
        for param in params:
            self.names.append(param.name)
        pass

    def read_experiments(self, filename=None, folder=None):
        data = ReadExperiments(filename=filename,folder=folder)
        # file = pd.read_csv(folder+"/"+filename)
        self.times.append(data.time.values)
        self.temperatures.append(data.temperature.values)
        self.dRho.append(data.dRho.values)
        self.Rho.append(data.Rho.values)

    def parameters(self):
        """

        :return:
        """
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        return simulations

    def evaluation(self):
        """

        :rtype: list with experimental observations
        """
        observations = [self.dRho,self.Rho]
        # observations = self.dRho
        return observations

    def objective_function(self,simulation,evaluation):
        """

        :param simulation: list with simulation results
        :param evaluation: list with experimental observations
        :return: float with value of the objective function
        """
        # Drho = rmse_multiple_files(evaluation,simulation) # 100 to give weight wrt rho
        objectivefunction = rmse_multiple_files(evaluation,simulation) # 100 to give weight wrt rho
        return objectivefunction

    def ode_solver(self,vector):
        """

        :param vector: vector of unknowns
        :return: list of results for simulation function
        """
        results_dRho = []
        results_Rho = []
        self.iternumber += 1
        os.makedirs(str(self.iternumber))

        for beta, temperature in zip(self.betas, self.temperatures):
            temperature = list(temperature)
            n_timesteps = len(temperature)
            PyroType = getattr(pyro,self.pyro_type)
            time_Exp = (temperature - temperature[0]) / (beta / 60)
            simulation = PyroType(temp_0=temperature[0],temp_end=temperature[-1], time=time_Exp, beta=beta, n_points=n_timesteps, isothermal=False)
            pyro.write_file_scheme(filename=self.scheme_file, vector=vector,param_names=self.names, folder=str(self.iternumber)+'/')
            simulation.react_reader(filename=self.scheme_file,folder=str(self.iternumber)+'/')
            simulation.param_reader(filename=self.scheme_file,folder=str(self.iternumber)+'/')
            simulation.solve_system()
            results_dRho.append(simulation.drho_solid)
            results_Rho.append(simulation.rho_solid)

        if self.keepFolders is False:
            shutil.rmtree(str(self.iternumber))

        return [results_dRho,results_Rho]
