from src.pyrolysis import PyrolysisCompetitive
import matplotlib.pyplot as plt
test = PyrolysisCompetitive(temp_0=373, temp_end=2000, time=None, beta=3, n_points=500)
# test.param_getter_opti([12,24,35,10,4,0.7,0.3],['E1','E2','E3','E4','A1','gamma1','gamma2'],'data_competing.json')
test.react_reader("data_competing_verification.json")
test.param_reader("data_competing_verification.json")
# test.react_writer("tests")
test.solve_system()
# test.plot_solid_density()
#
rho = test.get_density()
t = test.get_time()
T = test.get_temperature()
plt.plot(T,rho)


test = PyrolysisCompetitive(temp_0=373, temp_end=2000, time=None, beta=3, n_points=500)
# test.param_getter_opti([12,24,35,10,4,0.7,0.3],['E1','E2','E3','E4','A1','gamma1','gamma2'],'data_competing.json')
test.react_reader("data_optimized.json")
test.param_reader("data_optimized.json")
# test.react_writer("tests")
test.solve_system()
# test.plot_solid_density()
#
rho = test.get_density()
t = test.get_time()
T = test.get_temperature()


plt.plot(T,rho)
plt.show()
