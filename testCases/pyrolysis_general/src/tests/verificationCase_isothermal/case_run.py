from src.pyrolysis import PyrolysisCompetitive
from src.read_experiments import ReadExperiments
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
import numpy as np
from matplotlib.lines import Line2D
plt.style.use('seaborn-poster')

def create_dummy_line(**kwds):
    return Line2D([], [], **kwds)

cmap = plt.cm.Dark2.colors


figTime1, axTime1 = plt.subplots()

file = "initialization_data.csv"

filename = file
experiment = ReadExperiments(filename=filename, folder='.')



# for rate in rates:
test = PyrolysisCompetitive(temp_0=experiment.temperature.values[0], temp_end=experiment.temperature.values[-1],
                  time=experiment.time.values, beta=0, n_points=600, isothermal=True)
test.react_reader("data.json")
test.param_reader("data.json")
test.solve_system()

rho = test.get_rho_solid()
drho = test.get_drho_solid()
t = test.get_time()
T = test.get_temperature()

# axTime1.plot(t, rho)

r1_numeric, = axTime1.plot(t, test.z.y[0], 'o', color=cmap[0])
r2_numeric, = axTime1.plot(t, test.z.y[1], 'o', color=cmap[1])
r3_numeric, = axTime1.plot(t, test.z.y[2], 'o', color=cmap[2])


## Analytical solution
t_analytical = np.linspace(0,t[-1])
rhoA0 = test.rhoIni[0]
A1 = test.dict_params['A'][0]
E1 = test.dict_params['E'][0]
c1 = 10**A1*np.exp(-E1/(PyrolysisCompetitive.R*test.temperature[0]))
rhoA = rhoA0*np.exp(-c1*t_analytical)

A2 = test.dict_params['A'][1]
E2 = test.dict_params['E'][1]
c2 = 10**A2*np.exp(-E2/(PyrolysisCompetitive.R*test.temperature[0]))
g1 = 1-test.dict_params['g'][0][0]

rhoB = g1*c1*rhoA0/(c2-c1)*(np.exp(-c1*t_analytical)-np.exp(-c2*t_analytical))

g2 = 1-test.dict_params['g'][1][0]
rhoC = (g2*c2*g1*c1*rhoA0)/(c2-c1)*(-np.exp(-c1*t_analytical)/c1+np.exp(-c2*t_analytical)/c2)+g2*g1*rhoA0

r1_analytical, = axTime1.plot(t_analytical, rhoA, color=cmap[0])
r3_analytical, = axTime1.plot(t_analytical, rhoB, color=cmap[1])
r2_analytical, = axTime1.plot(t_analytical, rhoC, color=cmap[2])

axTime1.set_xlabel('Time (s)')
axTime1.set_ylabel(r'$\rho$ (%)')

# Create the legend
lines = [
    ('Analytical', {'color': 'k', 'linestyle': '-'}),
    ('Numerical', {'color': 'k', 'marker': 'o'}),
    ('A', {'color': cmap[0], 'linestyle': '-'}),
    ('B', {'color': cmap[1], 'linestyle': '-'}),
    ('C', {'color': cmap[2], 'linestyle': '-'}),
]

legend1 = plt.legend(
    # Line handles
    [create_dummy_line(**l[1]) for l in lines],
    # Line titles
    [l[0] for l in lines],
    loc='best',
    frameon = False
)

axTime1.add_artist(legend1)

plt.savefig('verification_iso.png')

plt.show()