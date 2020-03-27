from src.pyrolysis import PyrolysisCompetitive
import matplotlib.pyplot as plt

test = PyrolysisCompetitive()
test.react_reader("data_competing_verification.json")
test.param_reader("data_competing_verification.json")
test.solve_system(temp_0=373, temp_end=2000, time=None, beta=1, n_points=500)
rho = test.get_density()
t = test.get_time()
T = test.get_temperature()
plt.plot(T,rho)

test = PyrolysisCompetitive()
test.react_reader("data_competing_verification.json")
test.param_reader("data_competing_verification.json")
test.solve_system(temp_0=373, temp_end=2000, time=None, beta=10000, n_points=500)
rho = test.get_density()
t = test.get_time()
T = test.get_temperature()
plt.plot(T,rho)


test = PyrolysisCompetitive()
test.react_reader("data_optimized.json")
test.param_reader("data_optimized.json")
test.solve_system(temp_0=373, temp_end=2000, time=None, beta=1, n_points=500)
rho = test.get_density()
t = test.get_time()
T = test.get_temperature()
plt.plot(T,rho,'--')

test = PyrolysisCompetitive()
test.react_reader("data_optimized.json")
test.param_reader("data_optimized.json")
test.solve_system(temp_0=373, temp_end=2000, time=None, beta=10000, n_points=500)
rho = test.get_density()
t = test.get_time()
T = test.get_temperature()
plt.plot(T,rho,'--')


plt.show()
