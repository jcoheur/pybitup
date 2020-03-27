from src.pyrolysis import PyrolysisCompetitive
from src.read_experiments import ReadExperiments
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import expi

plt.style.use('seaborn-poster')

def create_dummy_line(**kwds):
    return Line2D([], [], **kwds)

cmap = plt.cm.Dark2.colors


figTime1, axTime1 = plt.subplots()

file = "initialization_data.csv"

filename = file
experiment = ReadExperiments(filename=filename, folder='.')
beta = 600 #K/min


# for rate in rates:
test = PyrolysisCompetitive(temp_0=experiment.temperature.values[0], temp_end=experiment.temperature.values[-1],
                  beta=beta, n_points=60)
test.react_reader("data.json")
test.param_reader("data.json")
test.solve_system()

rho = test.get_rho_solid()
drho = test.get_drho_solid()
t = test.get_time()
T = test.get_temperature()

# axTime1.plot(t, rho)

r1_numeric, = axTime1.plot(t, test.z.y[0], 'o', color=cmap[0])
# r2_numeric, = axTime1.plot(t, test.z.y[1], 'o', color=cmap[1])


## Analytical solution
t = np.linspace(0,t[-1], 200)
rhoA0 = test.rhoIni[0]
A1 = test.dict_params['A'][0]
E1 = test.dict_params['E'][0]
R = PyrolysisCompetitive.R
T0 = experiment.temperature.values[0]
beta = beta/60 # K/s

constant_part = ((10**A1)*(E1*expi(-E1/(R*T0))+R*T0*np.exp(-E1/(R*T0))))/(beta*R)

exponential_integral = expi(-E1/(R*(beta*t+T0)))
exponential = np.exp(-E1/(R*(beta*t+T0)))
first_summand = E1*exponential_integral
second_summand = R*(beta*t+T0)*exponential
sum_summands = first_summand+second_summand
product = 10**A1 * sum_summands
division = product/(beta*R)

time_dependant_part = (10**A1 * (E1*expi(-E1/(R*(beta*t+T0)))+R*(beta*t+T0)*np.exp(-E1/(R*(beta*t+T0)))))/(beta*R)

rhoA = rhoA0*np.exp(constant_part-time_dependant_part)

r1_analytical, = axTime1.plot(t, rhoA, color=cmap[0])

axTime1.set_xlabel('Time (s)')
axTime1.set_ylabel(r'$\rho$ (%)')

# # Create the legend
lines = [
    ('Analytical', {'color': cmap[0], 'linestyle': '-'}),
    ('Numerical', {'color': cmap[0], 'marker': 'o'}),
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

plt.savefig('verification_1_react_non_iso.png')
plt.show()
