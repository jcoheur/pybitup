from src.pyrolysis import PyrolysisCompetitive
from src.read_experiments import ReadExperiments
import matplotlib.pyplot as plt
from src.plotStyle import style

stylePlots = style()
plt.style.use(stylePlots)

figRho, axRho = plt.subplots()
figdRho, axdRho = plt.subplots()
figTime, axTime = plt.subplots()

axRho.set_xlabel('Temperature (K)')
axRho.set_ylabel('$\\rho/\\rho_{\\textrm{v}} (-)$')
axdRho.set_xlabel('Temperature (K)')
axdRho.set_ylabel('d$(\\rho/\\rho_{\\textrm{v}})$/d$T$ (mK$^{-1}$)')
axTime.set_xlabel('Temperature (K)')
axTime.set_ylabel('Time (s)')

rates = [366, 10]
filenames = ['Bessire_366Kmin.csv', "wong_data_10Kmin_resampled.csv"]
colors = ['C1','C0']
legendLocations = ('upper right', 'center right')
for rate,file,c,legendLoc in zip(rates,filenames, colors, legendLocations):
    filename = file
    experiment = ReadExperiments(filename=filename, folder='.')

    axRho.plot(experiment.temperature, experiment.Rho/100,'o', color=c,mfc='none',label='Exp. '+str(rate)+'K/min')
    axdRho.plot(experiment.temperature, experiment.dRho*10,'o',color=c,mfc='none',label='Exp. '+str(rate)+'K/min')
    axTime.plot(experiment.temperature, experiment.time,'o',color=c,mfc='none',label='Exp. '+str(rate)+'K/min')


# for rate in rates:
    test = PyrolysisCompetitive(temp_0=experiment.temperature.values[0], temp_end=experiment.temperature.values[-1], time=None, beta=rate, n_points=600)
    test.react_reader("data_optimized.json")
    test.param_reader("data_optimized.json")
    test.solve_system()

    rho = test.get_rho_solid()
    drho = test.get_drho_solid()
    t = test.get_time()
    T = test.get_temperature()

    axRho.plot(T,rho/100,color=c, label='Computed '+str(rate)+'K/min')
    axdRho.plot(T,drho*10,color=c, label='Computed '+str(rate)+'K/min')
    axTime.plot(T,t,color=c, label='Computed '+str(rate)+'K/min')

    # plot for the production/destruction of each term
    figProduction, axProduction = plt.subplots()
    labels = ('Reactant', '$\\textrm{Activation}_{\\textrm{slow}}$', '$\\textrm{Activation}_{\\textrm{fast}}$','Solid 1','Solid 2')
    for product, label in zip(test.z.y, labels):
        axProduction.plot(T, product/100, label=label)

    axProduction.set_xlabel('Temperature (K)')
    axProduction.set_ylabel('$\\rho/\\rho_{\\textrm{v}} (-)$')
    axProduction.legend(loc=legendLoc)
    figProduction.savefig(str(rate)+'.png')
    figProduction.savefig(str(rate)+'.eps')

axRho.legend(loc='best')
axdRho.legend(loc='best')
axTime.legend(loc='best')


# axdRho.ticklabel_format(style='sci', axis='y',scilimits=(0,10))

figRho.savefig('rho.png')
figdRho.savefig('drho.png')
figTime.savefig('time.png')

figRho.savefig('rho.eps')
figdRho.savefig('drho.eps')
figTime.savefig('time.eps')
plt.show()