from src.pyrolysis import PyrolysisParallel

import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
cmap = plt.cm.Dark2.colors


T_0=373
T_end=2000
b=20

test = PyrolysisParallel(temp_0=T_0, temp_end=T_end, time=None, beta=b, n_points=200)

test.react_reader(filename="data_parallel.json")
test.param_reader(filename="data_parallel.json")

# Numerical solution using 200 points
test.solve_system()
temperature = test.get_temperature()
plt.figure(1)
plt.plot(temperature, test.get_density())
plt.figure(2)
plt.plot(temperature, test.get_drhoSolid_dT())


# Compute analytical solution at 25 points
test.compute_time_temperature(temp_0=T_0, temp_end=T_end, time=None, beta=b, n_points=200)
test.compute_analytical_solution()
temperature = test.get_temperature()
plt.figure(1)
plt.plot(temperature, test.get_density(), 'o', color=cmap[0])
plt.figure(2)
plt.plot(temperature, test.get_drhoSolid_dT(), 'o', color=cmap[0])
plt.figure(3) 
plt.plot(temperature, test.num_grad['A',0])
plt.plot(temperature, test.anal_grad['A',0], 'o', color=cmap[0])
plt.figure(4) 
plt.plot(temperature, test.num_grad['A',1])
plt.plot(temperature, test.anal_grad['A',1], 'o', color=cmap[0])
plt.figure(5) 
plt.plot(temperature, test.num_grad['E',0])
plt.plot(temperature, test.anal_grad['E',0], 'o', color=cmap[0])
plt.figure(6) 
plt.plot(temperature, test.num_grad['E',1])
plt.plot(temperature, test.anal_grad['E',1], 'o', color=cmap[0])
plt.figure(7) 
plt.plot(temperature, test.num_grad['n',0])
plt.plot(temperature, test.anal_grad['n',0], 'o', color=cmap[0])
plt.figure(8) 
plt.plot(temperature, test.num_grad['n',1])
plt.plot(temperature, test.anal_grad['n',1], 'o', color=cmap[0])
plt.figure(9) 
plt.plot(temperature, test.num_grad['F',0])
plt.plot(temperature, test.anal_grad['F',0], 'o', color=cmap[0])
plt.figure(10) 
plt.plot(temperature, test.num_grad['F',1])
plt.plot(temperature, test.anal_grad['F',1], 'o', color=cmap[0])

# Create the legend
plt.figure(1)
plt.title('Mass loss')
lines = [
    ('Analytical', {'linestyle': '-'}),
    ('Numerical', {'color': cmap[0], 'marker': 'o'}),
]
legend1 = plt.legend(
    # Line titles
    [l[0] for l in lines],
    loc='best',
    frameon = False
)
plt.figure(2)
plt.title('Mass loss rate')
lines = [
    ('Analytical', {'linestyle': '-'}),
    ('Numerical', {'color': cmap[0], 'marker': 'o'}),
]
legend1 = plt.legend(
    # Line titles
    [l[0] for l in lines],
    loc='best',
    frameon = False
)

plt.show()
