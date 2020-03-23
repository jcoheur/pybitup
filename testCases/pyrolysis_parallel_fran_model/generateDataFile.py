import numpy as np 
import matplotlib.pyplot as plt
import random
import pandas as pd 


from pyrolysis_general.src.pyrolysis import PyrolysisParallel



T = np.linspace(300,1400, 101) 
T_0 = 300
tau = 6.1
time0 = (T - T_0)/(tau/60)

test = PyrolysisParallel(temp_0=T_0, temp_end=T[-1], time = time0, beta=tau, n_points=101)

test.react_reader(filename="data_parallel.json")
test.param_reader(filename="data_parallel.json")

test.solve_system()
plt.figure(1)
test.plot_solid_density()

plt.figure(2)
rho_solid = test.get_density()
drho_solid = test.get_drho_solid()
temperature = test.get_temperature()

std_y=0.00001
exp_std_y = std_y * 0.1 
n_runs = 3 

for j in range(n_runs):
    num_data = len(drho_solid)
    rn_data=np.zeros((1, num_data))
    for i in range(0, num_data):
        rn_data[0,i]=random.gauss(0, std_y)
        drho_solid_pert = drho_solid + rn_data[0,:]
            
            
    plt.plot(temperature, drho_solid_pert, '-o')

    myfile = {'time': time0, 'temperature': T, 'rho': rho_solid, 'dRho': drho_solid_pert, 'std_dRho': exp_std_y} 
    df = pd.DataFrame(myfile, columns=['time', 'temperature', 'rho', 'dRho', 'std_dRho'])


    #df.to_csv("fakePyroData_"+str(j)+".csv")

plt.show()
	

