from src.pyrolysis import PyrolysisCompetitiveSparse
import matplotlib.pyplot as plt
test = PyrolysisCompetitiveSparse()


# test.param_getter_opti([12,24,35,10,4,0.7,0.3],['E1','E2','E3','E4','A1','gamma1','gamma2'],'data_competing.json')
test.react_reader("data_competing_verification.json")
test.param_reader("data_competing_verification.json")
# test.react_writer("tests")
test.solve_system(temp_0=400, temp_end=1500, time=None, beta=1, n_points=600)
# test.plot_solid_density()
#
rho = test.get_density()
t = test.get_time()
T = test.get_temperature()
plt.plot(T,rho)
test.to_csv('SLOW_1.csv')


test.solve_system(temp_0=400, temp_end=1500, time=None, beta=10000, n_points=300)
# test.plot_solid_density()
#
rho = test.get_density()
t = test.get_time()
T = test.get_temperature()
plt.plot(T,rho)
test.to_csv('FAST_10000.csv')


plt.show()

