import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 


from pyrolysis_general.src.pyrolysis import PyrolysisParallel
from pybitup import bayesian_inference as bi


# Pyrolysis experiment parameters
# ---------------------------------- 
T_0 = 300
T = np.linspace(300,1200, 101) 
tau = 6.1
time0 = (T - T_0)/(tau/60)
test = PyrolysisParallel(temp_0=T_0, temp_end=T[-1], time = time0, beta=tau, n_points=101)

# Random parameters 
# ------------------
n_runs = 100 # Number of experimental runs 
generateRandParam = False 

if generateRandParam is True: 
    # Generate random parameter values 
    # ----------------------------------
    A0 = []
    E0 = [] 
    for j in range(n_runs):
        A0.append(random.gauss(16634,  1000))
        E0.append(random.gauss(113000,  1000))

    myfile = {'A0': A0, 'E0': E0}
    df = pd.DataFrame(myfile, columns=['A0', 'E0'])
    df.to_csv("data_files/random_param.csv")
else: 
    # Else, get random parameter values if there already exists in a random_param.csv file 
    # -------------------------------------------------------------------------------------
    param_values = pd.read_csv("data_files/random_param.csv", delimiter=',') 
    A0 = param_values["A0"].values
    E0 = param_values["E0"].values

# Generate random data from the random parameters 
# ---------------------------------------------

exp_std_y = 1e-6 # Experimental error (standard deviation)
ds_val = 2 # ds_val = 1 -> no downsample 
for j in range(n_runs):
    
    input_file_name = "reaction_scheme_2param.json"
    param_names = ["A0", "E0"]

    param_values = [A0[j], E0[j]] 
    bi.write_tmp_input_file(input_file_name, param_names, param_values)

    test.react_reader(filename="tmp_proc_0_"+input_file_name)
    test.param_reader(filename="tmp_proc_0_"+input_file_name)
    test.solve_system()

    plt.figure(1)
    rho_solid = test.get_density()
    drho_solid = test.get_drho_solid()
    temperature = test.get_temperature()

    drho_solid_pert = np.array(drho_solid) # Perturbed data 
    num_data = len(drho_solid)
    for i in range(0, num_data):
        rn_data=random.gauss(0, exp_std_y)
        drho_solid_pert[i] = drho_solid[i]  + rn_data
        if drho_solid_pert[i] < 0: 
            drho_solid_pert[i] = 0.0

    # If we discard the first values close to zero 
    # drho_solid_pert_f = drho_solid_pert[0:1]
    # drho_solid_pert_f = np.concatenate((drho_solid_pert_f, drho_solid_pert[35:-1]))
    # rho_solid_f = rho_solid[0:1]
    # rho_solid_f = np.concatenate((rho_solid_f, rho_solid[35:-1]))
    # temperature_f = temperature[0:1]
    # temperature_f = np.concatenate((temperature_f, temperature[35:-1])) 
    # time_f = time0[0:1]
    # time_f = np.concatenate((time_f, time0[35:-1]))

    # If we downsampel 
    drho_solid_pert_f = drho_solid_pert[0:-1:ds_val]
    rho_solid_f = rho_solid[0:-1:ds_val]
    temperature_f = temperature[0:-1:ds_val]
    time_f = time0[0:-1:ds_val]
    plt.plot(temperature_f, drho_solid_pert_f, '-o')

    # Save data file 
    myfile = {'time': time_f, 'temperature': temperature_f, 'rho': rho_solid_f, 'dRho': drho_solid_pert_f, 'std_dRho': exp_std_y} 
    df = pd.DataFrame(myfile, columns=['time', 'temperature', 'rho', 'dRho', 'std_dRho'])
    df.to_csv("data_files/fakePyroData_"+str(j)+".csv")

plt.show()
	

