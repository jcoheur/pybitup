import pyrolysis_general.src.pyrolysis as pyro
import numpy as np
from scipy.integrate import solve_ivp


# Implements model from Tranchard et al 2017 JAAP Kinetic analysis of the thermal decomposition of a carbon fibre-reinforced epoxy resin laminate

class PyrolysisEpoxy(pyro.Pyrolysis):

    def __init__(self, temp_0=373, temp_end=2000, time=None, beta=20, n_points=500):
        super().__init__(temp_0, temp_end, time, beta, n_points)

    def generate_rate(self, temperature=float("inf")):
        k = []
        for idx in range(0, self.n_reactions):
            k.append(10 ** self.dict_params['A'][idx] *
                     np.exp(-self.dict_params['E'][idx] / (self.R * temperature)))
        return k

    def pyro_rates(self, z, t=0, T0=float("inf"), betaKs=0.3333):
        """ Compute the pyrolysis reaction rates at a given temperature """
        temperature = T0 + betaKs * t
        tol = 10**-4
        dchidt = []
        sum_z = sum(z)
        if abs(1-sum_z)<tol:
            dchidt = np.zeros(2)
        else:
            k = self.generate_rate(temperature)
            for idx in range(0,self.n_reactions):
                dchidt.append(k[idx]*(1-sum_z)**self.dict_params['n'][idx]*(1+self.dict_params["Kcat"][idx]*z[idx]))

        return dchidt

    def solve_system(self, isothermal=False):
        """ Solve the linear system of equation of pyrolysis reaction.
              By default, only Radau method for solving the initial value problem. """



        paramStep = 1  # for the max step in solver, adjust if needed
        max_step = paramStep / self.betaKs * 100
        temp_0 = self.temperature[0]

        y0 = np.zeros(self.n_reactions)
        self.z = solve_ivp(fun=lambda t, z: self.pyro_rates(z, t, temp_0, self.betaKs),
                               t_span=(0, self.time[-1]),
                               y0=y0,
                               t_eval=self.time,
                               method="Radau", max_step=max_step, rtol=1E-5)

        self.time = self.z.t
        self.temperature = self.time * self.betaKs + temp_0

        # Compute the density evolution from chi
        self.rho_solid = np.zeros(len(self.time))
        percent_evo_sum = np.zeros(len(self.time))
        for chi,F  in zip(self.z.y,self.dict_params["F"]):
            percent_evo = chi*F
            percent_evo_sum += percent_evo

        self.rho_solid = self.rhoIni[0]-percent_evo_sum

        self.drho_solid = np.gradient(-self.rho_solid, self.temperature)
