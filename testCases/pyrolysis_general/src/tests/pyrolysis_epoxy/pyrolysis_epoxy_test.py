import pyrolysis_general.src.pyrolysis_epoxy as pyro
import matplotlib.pyplot as plt


rates = [5,10,20,50,100]
for rate in rates:
    test = pyro.PyrolysisEpoxy(temp_0=308, temp_end=1373, time=None,
                  beta=rate, n_points=600)

    test.react_reader(filename="data_epoxy.json")
    test.param_reader(filename="data_epoxy.json")
    test.solve_system()
    T = test.get_temperature()
    rho = test.get_rho_solid()
    drho = test.get_drho_solid()
<<<<<<< HEAD
    
    plt.figure(1)
    plt.plot(T,rho)
    
    plt.figure(2)
    plt.plot(T,drho)

=======
    plt.plot(T,rho)
>>>>>>> master
# plt.plot(test.temperature, test.z.y[0])
# plt.plot(test.temperature, test.z.y[1])
# plt.plot(test.temperature, test.z.y[1]+test.z.y[0])
# plt.savefig('1.png')
plt.show()