import matplotlib.pyplot as plt
import spotpy
from src.optimizer import spotpy_setup
import pandas as pd
import src.pyrolysis as pyro


results=[]
folder = "."
# filesCSV = [f for f in os.listdir(folder) if f.endswith('.csv')]
# filesCSV = ['Bessire_366Kmin.csv']
# filesCSV = ['Wong_10Kmin.csv']
filesCSV = ['Bessire_366Kmin.csv','wong_data_10Kmin_resampled.csv']

params = [
            spotpy.parameter.Uniform('E1',low=10E3 , high=65E3,  optguess=47000),
            spotpy.parameter.Uniform('E2',low=80E3, high=180E3, optguess=150000),
            spotpy.parameter.Uniform('E3', low=20E3 , high=80E3, optguess=50000),
            spotpy.parameter.Uniform('E4', low=10E3, high=60E3, optguess=51000),
            spotpy.parameter.Uniform('A1', low=2, high=6, optguess=3.9),
            spotpy.parameter.Uniform('A2', low=10, high=22, optguess=15.6),
            spotpy.parameter.Uniform('A3', low=0.2, high=4, optguess=0.4),
            spotpy.parameter.Uniform('A4', low=0.5, high=5, optguess=3.1),
            spotpy.parameter.Uniform('gamma1',  low=0, high=0.01, optguess=0.1),
            spotpy.parameter.Uniform('gamma2', low=0, high=0.1, optguess=0.1),
            spotpy.parameter.Uniform('gamma3',  low=0, high=0.25, optguess=0.1),
            spotpy.parameter.Uniform('gamma4', low=0, high=0.35, optguess=0.1),
]

spotpy_setup = spotpy_setup(files=filesCSV, params=params,
                            folder=folder, scheme_file="data.json",
                            pyro_type="PyrolysisCompetitive", isothermal=False)
rep=10000

sampler=spotpy.algorithms.sceua(spotpy_setup, dbname='SCEUA', dbformat='ram', save_sim=False,
                                alt_objfun="rmse_multiple_files")
sampler.sample(rep,ngs=12)
results.append(sampler.getdata())
names = spotpy.analyser.get_parameternames(sampler.getdata())
best = spotpy.analyser.get_best_parameterset(sampler.getdata())

bestResults = [x for x in best[0]]

resultCSV = pd.DataFrame.from_dict({"Params":names, "Value":bestResults})
resultCSV.to_csv(folder+'/'+"results_"+str(rep))

pyro.replace_results(bestResults, names, 'data.json.template', 'data_optimized.json')