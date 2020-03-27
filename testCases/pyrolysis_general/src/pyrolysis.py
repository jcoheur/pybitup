import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy import special
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pandas as pd
import fileinput
import shutil
from jsmin import jsmin
import os


class Pyrolysis(ABC):
    R = 8.314

    def __init__(self, temp_0, temp_end, time, beta, n_points):
        # self.filename = scheme_file
        # with open(self.filename) as f:
        #     self.data = json.load(f)
        self.param_names = None
        self.dict_params = dict()
        self.rho_solid = 0
        self.drhoSolid_dT = 0
        self.drhoSolid_dt = 0
        self.drho_solid = 0
        self.drho_species_density = 0

        # Gradients with respect to parameters 
        self.anal_grad = {}
        self.num_grad = {}

        # Convert beta in K/min to betaKs in K/sec
        self.betaKs = beta / 60

        # Compute temperature and time
        self.compute_time_temperature(temp_0, temp_end, time, beta, n_points)

    def compute_time_temperature(self, temp_0=373, temp_end=2000, time=None, beta=20, n_points=500):
        """ Compute temperature as a function of time based on the given heating rate and 
        intial temperature temp_0. If no time vector is provided, a time vector of size n_points 
        is computed from temp_0 to temp_end"""

        if time is None:
            self.time = np.linspace(0, (temp_end - temp_0) / self.betaKs, n_points)
        else:
            self.time = time

        self.temperature = self.time * self.betaKs + temp_0

    def react_reader(self, filename, folder='./'):
        """ Read the reaction processes from input file """
        self.dict_params = dict()
        with open(folder + filename) as f:
            minified = jsmin(f.read())
            data = json.loads(minified)
        self.solids = data["solids"]
        self.species = data["species"]

        # self.gas = self.data["gas"]
        self.reactions = data["reactions"]
        self.n_reactions = len(self.reactions)
        self.rhoIni = data["rhoIni"]
        reactants = []
        rhsList = []  # this is the list of all the products of one reaction (right hand side)

        # Build the list of reactants and products (contained in rhs)
        for reaction in self.reactions:
            keys = reaction.keys()
            for key in keys:
                if key in self.solids:
                    reactants.append(key)
                    rhsList.append(reaction[key])

        self.rhs = rhsList
        products = []
        solidProduct = []
        gasesProduct = []

        for rhs in rhsList:
            rhsSplit = rhs.strip().replace(" ", "").split('+')
            gases = rhsSplit
            products.append(rhsSplit)
            for product in rhsSplit:
                if product in self.solids:
                    solidProduct.append(product)
                    gases.remove(product)
            gasesProduct.append(gases)
        self.unique_gases = list(set([item for sublist in gasesProduct for item in
                                      sublist]))  # flattens list, and gets unique gases to make gas list
        self.n_gas_species = len(self.unique_gases)

        # # Check for missing solids
        # solid_check = list(set(solidProduct + reactants))
        # if len(solid_check) is not len(self.solids):
        #     raise ValueError("There is a problem with the solids")

        self.solid_product = solidProduct
        self.solid_reactant = reactants
        self.gas_product = gasesProduct
        self.n_solids = len(self.solids)
        self.g_sol = []

        return 0

    def param_reader(self, filename, folder='./'):
        """ Read the parameter values after the reaction reader """
        self.param_names = None
        with open(folder + filename) as f:
            data = json.load(f)
        parameters = data["parameters"]
        self.param_names = parameters[0].keys()
        for react in parameters:
            for key in self.param_names:
                if key in self.dict_params:
                    self.dict_params[key].append(react[key])
                else:
                    self.dict_params[key] = []
                    self.dict_params[key].append(react[key])

        if 'g' in self.dict_params.keys():
            for g_react in self.dict_params['g']:
                g_sol_temp = sum(g_react)
                self.g_sol.append(1-g_sol_temp)

        # Make a mapping list between dict['g'] and the index of the corresponding species
        self.mapSpeciesListIndex = []
        for product_list in self.gas_product: 
            c_map = []
            for product in product_list: 
                for c_idx, c_species in enumerate(self.species): 
                    if product == c_species: 
                        c_map.append(c_idx) 
                        break 
            self.mapSpeciesListIndex.append(c_map)

        return 0

    # def param_getter_opti(self,vector,param_names,filename, symbolleft='(', symbolright=')', folder = './'):
    #     with open(filename+".template", 'r') as fileread:
    #         with open(folder+filename, 'w+') as filewrite:
    #             for line in fileread:
    #                 line_new = line
    #                 for val, param in zip(vector, param_names):
    #                     line_new = line_new.replace(symbolleft+param+symbolright, str(val))
    #                 line = line_new
    #                 filewrite.write(line)

    def get_param(self):
        return self.dict_params

    def get_unique_gases(self):
        return self.unique_gases

    def get_solids(self):
        return self.solids

    def get_num_reactions(self):
        return self.n_reactions

    def get_density(self):
        return self.rho_solid

    get_rho_solid = get_density

    def get_drhoSolid_dT(self): 
        return self.drhoSolid_dT
	
    def get_drhoSolid_dt(self): 
        return self.drhoSolid_dt
	
    def get_drho_solid(self):
        """Is the same as get_drhoSolid_dT
        Need to be changed later to avoid ambiguity in the derivative"""
        return self.drho_solid

    def get_drho_species_density(self): 
        return self.drho_species_density
	
    def get_time(self):
        return self.time

    def get_temperature(self):
        return self.temperature

    def get_drho_species_dparam(self):
        return self.anal_grad

    @abstractmethod
    def pyro_rates(self):
        pass

    @abstractmethod
    def solve_system(self):
        pass

    def react_writer(self, filename):
        data = {}
        data["rhoIni"] = self.rhoIni
        data["solids"] = self.solids

        reactions_rebuilt = []
        for idx in range(0, self.n_reactions):
            react_dict = {}
            react_dict[self.solid_reactant[idx]] = self.rhs[idx]
            # react_dict['A'] = self.A[idx]
            # react_dict['E'] = self.E[idx]
            # react_dict['g'] = self.gammas[idx]
            reactions_rebuilt.append(react_dict)
        data["reactions"] = reactions_rebuilt

        param_rebuilt = []
        for idx in range(0, self.n_reactions):
            param_dict = {}
            for parameter in self.param_names:
                param_dict[parameter] = self.dict_params[parameter][idx]
            param_rebuilt.append(param_dict)
        data["parameters"] = param_rebuilt

        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=2)

    def plot_solid_density(self):
        """ Plot total solid density as a function of time """

        plt.plot(self.temperature, self.rho_solid)
        plt.show()

    def plot_var(self, var_name):
        pass

    def to_csv(self, name):
        file_save = pd.DataFrame(data=None, columns=[], index=None)
        file_save['time'] = self.time
        file_save['temperature'] = self.temperature
        file_save['rho'] = self.rho_solid
        file_save['dRho'] = self.drho_solid
        file_save.to_csv(name)


class PyrolysisParallel(Pyrolysis):
    def __init__(self, temp_0=373, temp_end=2000, time=None, beta=20, n_points=500, isothermal=False):
        super().__init__(temp_0, temp_end, time, beta, n_points)
        self.isothermal = isothermal

    def generate_rate(self, temperature=float("inf")):
        k = []
        for idx in range(0,self.n_reactions):
            # k.append(10 ** self.dict_params['A'][idx] *
            #          np.exp(-self.dict_params['E'][idx]/(self.R * temperature)))
             k.append(self.dict_params['A'][idx] *
                     np.exp(-self.dict_params['E'][idx]/(self.R * temperature)))
        return k

    def pyro_rates(self, z=0, t=0, T0=float("inf"), betaKs=0.3333):
        """ Compute the pyrolysis reaction rates at a given temperature """
        temperature = T0 + betaKs * t
        k = self.generate_rate(temperature)
        dchidt = []
        for idx in range(0, self.n_reactions):
            if (1 - z[idx]) < 1e-5:
                z[idx] = 1
            dchidt.append(k[idx] * (1 - z[idx]) ** self.dict_params['n'][idx])
        return dchidt

    def solve_system(self):
        """ Solve the linear system of equation of pyrolysis reaction.
              By default, only Radau method for solving the initial value problem. """

        paramStep = 5  # for the max step in solver, adjust if needed
        max_step = paramStep / self.betaKs * 100
        temp_0 = self.temperature[0]

        y0 = np.zeros(self.n_reactions)
        self.z = solve_ivp(fun=lambda t, z: self.pyro_rates(z, t, temp_0, self.betaKs),
                           t_span=(0, self.time[-1]),
                           y0=y0,
                           t_eval=self.time,
                           method="Radau", max_step=max_step, rtol=1E-5)

        self.time = self.z.t

        # Compute the density evolution from chi
        self.rho_solid = np.zeros(len(self.time))
        percent_evo_sum = np.zeros(len(self.time))
        for chi, F in zip(self.z.y, self.dict_params["F"]):
            percent_evo = chi * F
            percent_evo_sum += percent_evo
					
        # Mass loss and mass loss rate
        self.rho_solid = self.rhoIni * (1 - percent_evo_sum)
        self.drho_solid = np.gradient(-self.rho_solid, self.temperature)

        self.drhoSolid_dt = np.gradient(-self.rho_solid, self.time)
        self.drhoSolid_dT = np.gradient(-self.rho_solid, self.temperature)
		
    def compute_analytical_solution(self):
        """ For parallel reactions, there exists an analytical solution (see Torres, Coheur, NASA TM 2018).
        The solution is implemented here. Sensitivities (derivatives with respect to parameters) are 
        also implemented and computed. This function does the same as solve_system. """
		
        tau = self.betaKs  
        R = Pyrolysis.R
        T = self.temperature
        xi_init = 0

		# Exponential integral function, and lambda functions for the analytical solution 
        self.ei = special.expi
        self.fun_pi_j = lambda A, exp_minus_E_over_RT, n, F, xi_T : F * (1 - xi_T)**n * (A/tau) * exp_minus_E_over_RT 
        self.C = {}
        self.xi_T = {}

        # Initialiaztion
        percent_evo_sum = np.zeros(len(self.time))
        pi_j_idx = {}
        pi_i = np.zeros((self.n_gas_species, len(self.time)))
        self.d_pi_i = {}

        pi_tot = np.zeros(len(self.time))
		
		# Compute solution for each parallel reaction and the sensitivities
        for idx in range(0,self.n_reactions):
            n = self.dict_params['n'][idx]
            #A = 10 ** self.dict_params['A'][idx]
            A = self.dict_params['A'][idx]
            E = self.dict_params['E'][idx]
            F = self.dict_params["F"][idx] 
			
            # Analytical solution 
            self.C[idx], self.xi_T[idx] = self.compute_advancement_coeff(A, E, n, xi_init, tau, T)
            percent_evo_sum +=self.xi_T[idx]*self.dict_params["F"][idx]
            
            pi_j_idx[idx] =  self.dict_params["F"][idx] * (1 - self.xi_T[idx])**n * (A/tau) * np.exp(-E / (R * self.temperature)) # (kg/K)
            pi_tot +=   pi_j_idx[idx] # (kg/K)
            #pi_tot +=  self.fun_pi_j(A, exp_minus_E_over_RT, n, F, self.xi_T[idx]) # Idem pi_j
            #pi_idx = self.fun_pi_j(A, exp_minus_E_over_RT, n, F, self.xi_T[idx]) # Idem pi_j_idx 

            for i, index_species in enumerate(self.mapSpeciesListIndex[idx]): 
                pi_i[index_species, :] += self.dict_params['g'][idx][i] * pi_j_idx[idx] 
		

        #     # Numerical sensitivities (use for verifying that solutions is correclty implemented) 
        #     exp_minus_E_over_RT = np.exp(-E/R/T) 
        #     self.compute_numerical_sensitivities(A, E, n, F, xi_init, tau, T, exp_minus_E_over_RT, self.xi_T[idx], pi_j_idx[idx], idx)	
        # # Analytical sensitivities
        # self.compute_analytical_sensitivities()
        # # Graphical comparison numerical with analytical 
        # for num_fig, key in enumerate(self.anal_grad.keys()): 
        #     plt.figure(np.floor(num_fig/4))
        #     # num_grad is multiplied by tau to have units in (kg/s)
        #     plt.plot(self.num_grad[key]*tau, '*')
        #     plt.plot(self.anal_grad[key], label=key)
        #     plt.legend()
        # plt.show()

		# Mass loss and mass loss rate
        self.rho_solid = self.rhoIni*(1-percent_evo_sum)
        self.drhoSolid_dT = self.rhoIni * pi_tot
        self.drhoSolid_dt = self.drhoSolid_dT * tau 
        self.drho_solid = self.drhoSolid_dT # to be removed (it's equal as drhoSolid_dT, but I keep for Fran for compatibility)
        
        self.drho_species_density = pi_i

    def compute_advancement_coeff(self, A, E, n, xi_init, tau, T):
        """ Compute the advancement of reaction coefficient analytically. """
        R = self.R
        E_over_R = E/R
        exp_minus_E_over_RT = np.exp(-E_over_R/T)
        expint_minus_E_over_RT =  self.ei(-E_over_R/T)
			
        C = (1 - xi_init)**(1 - n)/(1 - n) + (A / tau) * T[0] * exp_minus_E_over_RT[0] \
            + expint_minus_E_over_RT[0] * E * (A / tau) / R
        xi_T = 1 - ((1-n) * (-(A / tau) * T * exp_minus_E_over_RT \
            - expint_minus_E_over_RT * E * (A / tau) / R + C))**(1/(1-n))
        
        return C, xi_T

    def compute_numerical_sensitivities(self, A, E, n, F, xi_init, tau, T, exp_minus_E_over_RT, xi_T, pi_idx, idx): 
        """ Computation of sensitivities (derivatives of gas production with respect to parameters) using finite differences. """ 

        Ap = A + A/1000
        Cp, xi_Tp = self.compute_advancement_coeff(Ap, E, n, xi_init, tau, T)
        pi_j_p =   self.fun_pi_j(Ap, exp_minus_E_over_RT, n, F, xi_Tp)
        self.num_grad['A_'+str(idx)] = (self.rhoIni*(pi_j_p-pi_idx)/(A/1000))
        
        Ep = E + E/1000
        Cp, xi_Tp = self.compute_advancement_coeff(A, Ep, n, xi_init, tau, T)
        exp_minus_Ep_over_RT = np.exp(-Ep/self.R/T)
        pi_j_p =  self.fun_pi_j(A, exp_minus_Ep_over_RT, n, F, xi_Tp)
        self.num_grad['E_'+str(idx)] = (self.rhoIni*(pi_j_p-pi_idx)/(E/1000))
        
        n_p = n + n/1000
        Cp, xi_Tp = self.compute_advancement_coeff(A, E, n_p, xi_init, tau, T)
        pi_j_p =  self.fun_pi_j(A, exp_minus_E_over_RT, n_p, F, xi_Tp)
        self.num_grad['n_'+str(idx)] = (self.rhoIni*(pi_j_p-pi_idx)/(n/1000))
        
        F_p = F + F/1000
        pi_j_p =  self.fun_pi_j(A, exp_minus_E_over_RT, n, F_p, xi_T)
        self.num_grad['F_'+str(idx)] = (self.rhoIni*(pi_j_p-pi_idx)/(F/1000))

        for i, index_species in enumerate(self.mapSpeciesListIndex[idx]): 
            g_p = self.dict_params['g'][idx][i] + self.dict_params['g'][idx][i]/1000
            pi_i = self.dict_params['g'][idx][i] * pi_idx
            pi_i_p = g_p * pi_idx
            self.num_grad['g_'+str(idx)+str(index_species)] = (self.rhoIni*(pi_i_p-pi_i)/(self.dict_params['g'][idx][i]/1000))

			

    def compute_analytical_sensitivities(self): 
        """ Compute the sentivities  (derivatives of gas production with respect to parameters) analytically. """
        """ Units of gas production is in kg/s ! (not kg/K, or you must multiply by tau) """

        R = self.R
        tau = self.betaKs
        T = self.temperature

        # Initialise
        for idx in range(0,self.n_gas_species):
            for index_react in range(self.n_reactions): 
                self.d_pi_i[idx, 'A_'+str(index_react)] = T*0
                self.d_pi_i[idx, 'E_'+str(index_react)] = T*0
                self.d_pi_i[idx, 'n_'+str(index_react)] = T*0
                for index_species in range(self.n_gas_species):
                    self.d_pi_i[idx, 'g_'+str(index_react)+str(index_species)] = T*0

        for idx in range(0,self.n_reactions):
            #A = 10 ** self.dict_params['A'][idx]
            A = self.dict_params['A'][idx]
            E = self.dict_params['E'][idx]
            n = self.dict_params['n'][idx]
            F = self.dict_params["F"][idx] 
           
            E_over_R = E/R
            A_over_tau = A/tau 
            exp_minus_E_over_RT = np.exp(-E_over_R/T)
            expint_minus_E_over_RT =  self.ei(-E_over_R/T)
            
            dC_dA = T[0]/tau*exp_minus_E_over_RT[0] + expint_minus_E_over_RT[0]*E_over_R/tau 
            dxik_dAk = -((1-n)*(-A_over_tau*T*exp_minus_E_over_RT-A_over_tau*E_over_R*expint_minus_E_over_RT+self.C[idx]))**(n/(1-n)) \
                * (-T/tau*exp_minus_E_over_RT - E_over_R/tau*expint_minus_E_over_RT+dC_dA)
            dpi_j_dA_k = n*(1-self.xi_T[idx])**(n-1)*(-dxik_dAk)*A*exp_minus_E_over_RT + (1-self.xi_T[idx])**n * exp_minus_E_over_RT
            self.anal_grad['A_'+str(idx)] = dpi_j_dA_k*self.rhoIni*F

            dC_dE = A_over_tau/R*expint_minus_E_over_RT[0]
            dxik_dEk = -((1-n)*(-A_over_tau*T*exp_minus_E_over_RT-A_over_tau*E_over_R*expint_minus_E_over_RT+self.C[idx]))**(n/(1-n)) \
                *(-A_over_tau/R*expint_minus_E_over_RT+dC_dE)
            # dpi_j_dE_k = (1-self.xi_T[idx])**n * A * exp_minus_E_over_RT*((n * (-dxik_dEk))/(1-self.xi_T[idx]) - 1/(R*T))
            dpi_j_dE_k = n * (1-self.xi_T[idx])**(n-1) *  (-dxik_dEk) * A * exp_minus_E_over_RT - (1-self.xi_T[idx])**n * A/(R*T) * exp_minus_E_over_RT

            self.anal_grad['E_'+str(idx)] = dpi_j_dE_k*self.rhoIni*F

            B = -A_over_tau*(T*exp_minus_E_over_RT + E_over_R*expint_minus_E_over_RT) + self.C[idx]
            dC_dn = (1-self.xi_T[idx][0])**(1-n)/(1-n)**2 - (1-self.xi_T[idx][0])**(1-n)*np.log(1-self.xi_T[idx][0])/(1-n)
            dxi_dn = -((1-n)*B)**(1/(1-n))*((-B+(1-n)*dC_dn)/((1-n)**2*B)+np.log((1-n)*B)/(1-n)**2)
            # dpi_j_dn_k = A*exp_minus_E_over_RT*(1-self.xi_T[idx])**n*(n*dxi_dn/(self.xi_T[idx]-1) + np.log(1-self.xi_T[idx]))
            
            # We compute the limit of (1-xi)*log(1-xi) when xi -> 1. Its value is equal to zero but we need to set 
            # it directly to avoid computation of 0*log(0)
            # one_m_xi_log_1_m_xi = (1-self.xi_T[idx][it]) * np.log(1-self.xi_T[idx][it])
            one_m_xi_log_1_m_xi = 0*T
            for it, xi in enumerate(self.xi_T[idx]):
                if xi <= (1-1e-5): 
                   one_m_xi_log_1_m_xi[it] = (1-self.xi_T[idx][it]) * np.log(1-self.xi_T[idx][it])
                else: 
                    one_m_xi_log_1_m_xi[it] = 0
            
            dpi_j_dn_k = A*exp_minus_E_over_RT*(1-self.xi_T[idx])**(n-1)*(-n*dxi_dn + one_m_xi_log_1_m_xi)
            self.anal_grad['n_'+str(idx)] = dpi_j_dn_k*self.rhoIni*F

            pi_idx = F * (1 - self.xi_T[idx])**n * A * exp_minus_E_over_RT 
            self.anal_grad['F_'+str(idx)] = self.rhoIni*pi_idx/F
                
            for i, index_species in enumerate(self.mapSpeciesListIndex[idx]):
                self.d_pi_i[index_species, 'A_'+str(idx)] = self.anal_grad['A_'+str(idx)]*self.dict_params['g'][idx][i]/tau
                self.d_pi_i[index_species, 'E_'+str(idx)] = self.anal_grad['E_'+str(idx)]*self.dict_params['g'][idx][i]/tau
                self.d_pi_i[index_species, 'n_'+str(idx)] = self.anal_grad['n_'+str(idx)]*self.dict_params['g'][idx][i]/tau
                self.d_pi_i[index_species, 'g_'+str(idx)+str(index_species)] = self.rhoIni*pi_idx/tau


class PyrolysisCompetitive(Pyrolysis):
    def __init__(self, temp_0=373, temp_end=2000, time=None, beta=20, n_points=500, isothermal=False):
        super().__init__(temp_0, temp_end, time, beta, n_points)
        self.isothermal = isothermal

    def generate_matrix(self, temperature=float("inf")):
        """ Build the coefficient matrix A for the linear pyrolysis reactions.
        See Eq. (3.6) in Torres et al., NASA TM) """

        k_loss = np.zeros((self.n_solids, self.n_solids))
        k_gain = np.zeros((self.n_solids, self.n_solids))

        for solid in range(0, self.n_solids):
            for reaction in range(0, self.n_reactions):

                # Compute the loss in reactants
                if self.solids[solid] == self.solid_reactant[reaction]:
                    k_loss[solid][solid] += (10 ** self.dict_params['A'][reaction]) * np.exp(
                        -self.dict_params['E'][reaction] / (self.R * temperature))

                # Compute the gain in (solid) products
                if self.solids[solid] == self.solid_product[reaction]:
                    idx_reactant = self.solids.index(
                        self.solid_reactant[reaction])  # finds which reactant produces this solid
                    k_gain[solid][idx_reactant] += self.g_sol[reaction] * (
                                10 ** self.dict_params['A'][reaction]) * np.exp(
                        -self.dict_params['E'][reaction] / (self.R * temperature))  # k of Arrhenius
        return -k_loss + k_gain

    def pyro_rates(self, z, t, T0, betaKs):
        """ Compute the pyrolysis reaction rates at a given temperature """

        temperature = T0 + betaKs * t
        drhodt = np.dot(self.generate_matrix(temperature), z)
        return drhodt

    def solve_system(self):
        """ Solve the linear system of equation of pyrolysis reaction.
        By default, only Radau method for solving the initial value problem. """

        paramStep = 5  # for the max step in solver, adjust if needed
        max_step = paramStep / self.betaKs * 100
        temp_0 = self.temperature[0]

        # Solve the system
        if self.isothermal:
            self.z = solve_ivp(fun=lambda t, z: self.pyro_rates(z, t, temp_0, self.betaKs), t_span=(0, self.time[-1]),
                               y0=self.rhoIni,
                               t_eval=self.time, rtol=1E-5)
        else:
            self.z = solve_ivp(fun=lambda t, z: self.pyro_rates(z, t, temp_0, self.betaKs), t_span=(0, self.time[-1]),
                               y0=self.rhoIni,
                               t_eval=self.time,
                               method="Radau", max_step=max_step, rtol=1E-5)

        self.time = self.z.t

        # Compute the total solid density
        self.rho_solid = np.zeros(len(self.time))
        for rho in self.z.y:
            self.rho_solid += rho

        # Check that dimensions are consistent
        if len(self.rho_solid) != len(self.time):
            self.rho_solid = np.zeros(len(self.time))
            print("Inconsistency in rho_solid and self.time dimensions. rho_solid set to zero.")

        # Compute total solid density gradient
        self.drho_solid = np.gradient(-self.rho_solid, self.temperature)

        return 0


class PyrolysisCompetitiveSparse(PyrolysisCompetitive):  # TODO: implement it...
    def __init__(self):
        super().__init__()

    def generate_matrix(self, temperature=float("inf")):
        """ Build the coefficient matrix A for the linear pyrolysis reactions.
        See Eq. (3.6) in Torres et al., NASA TM) """

        k_loss = np.zeros((self.n_solids, self.n_solids))
        k_gain = np.zeros((self.n_solids, self.n_solids))

        for solid in range(0, self.n_solids):
            for reaction in range(0, self.n_reactions):

                # Compute the loss in reactants
                if self.solids[solid] == self.solid_reactant[reaction]:
                    k_loss[solid][solid] += (10 ** self.dict_params['A'][reaction]) * np.exp(
                        -self.dict_params['E'][reaction] / (Pyrolysis.R * temperature))

                # Compute the gain in (solid) products
                if self.solids[solid] == self.solid_product[reaction]:
                    idx_reactant = self.solids.index(
                        self.solid_reactant[reaction])  # finds which reactant produces this solid
                    k_gain[solid][idx_reactant] += self.g_sol[reaction] * (
                                10 ** self.dict_params['A'][reaction]) * np.exp(
                        -self.dict_params['E'][reaction] / (Pyrolysis.R * temperature))  # k of Arrhenius

        return -k_loss + k_gain

    def pyro_rates(self, z, t, T0, betaKs):
        """ Compute the pyrolysis reaction rates at a given temperature """
        temperature = T0 + betaKs * t
        drhodt = np.dot(self.generate_matrix(temperature), z)
        return drhodt


def write_file_scheme(vector, param_names, filename, symbolleft='(', symbolright=')', folder='./'):
    with open(filename + ".template", 'r') as fileread:
        with open(folder + filename, 'w+') as filewrite:
            for line in fileread:
                line_new = line
                for val, param in zip(vector, param_names):
                    line_new = line_new.replace(symbolleft + param + symbolright, str(val))
                line = line_new
                filewrite.write(line)


def replace_results(vector, param_names, filename_template, filename_out, symbolleft='(', symbolright=')'):
    with open(filename_template, 'r') as fileread:
        with open(filename_out, 'w+') as filewrite:
            for line in fileread:
                line_new = line
                for val, param in zip(vector, param_names):
                    line_new = line_new.replace(symbolleft + param + symbolright, str(val))
                line = line_new
                filewrite.write(line)
