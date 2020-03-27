import matplotlib.pyplot as plt
import spotpy
from src.optimizer import spotpy_setup
import pandas as pd
import src.pyrolysis as pyro

results=[]
folder = "."

# filesCSV = ["test_20Kmin.csv", "test_1Kmin.csv"]
filesCSV = ["test_3Kmin.csv"]

params = [
            spotpy.parameter.Uniform('E1',low=70E3 , high=150E3,  optguess=70E3),
            spotpy.parameter.Uniform('E2',low=90E3, high=170E3, optguess=10E4),
            spotpy.parameter.Uniform('A1', low=1, high=4, optguess=3.9),
            spotpy.parameter.Uniform('A2', low=1, high=4, optguess=3.9),
    # spotpy.parameter.Uniform('gamma1', low=0.05, high=0.3, optguess=0.1),
            # spotpy.parameter.Uniform('gamma2', low=0.1, high=0.15, optguess=0.1),
]

spotpy_setup = spotpy_setup(files=filesCSV, params=params,
                            folder=folder, scheme_file="data_competing.json",
                            pyro_type="PyrolysisCompetitive", keepFolders=False)
rep=500

# sampler = spotpy.algorithms.sceua(spotpy_setup, dbname='SCEUA', dbformat='ram',alt_objfun="rmse_multiple_files", save_sim=False)
sampler = spotpy.algorithms.sceua(spotpy_setup, dbname='SCEUA', dbformat='ram',alt_objfun="rmse_multiple_files", save_sim=False)
sampler.sample(rep,ngs=8)
results.append(sampler.getdata())
names = spotpy.analyser.get_parameternames(sampler.getdata())
best = spotpy.analyser.get_best_parameterset(sampler.getdata())

spotpy_setup.plotBest(bestResultVector=best[0])
bestResults = [x for x in best[0]]

resultCSV = pd.DataFrame.from_dict({"Params":names, "Value":bestResults})
resultCSV.to_csv(folder+'/'+"results_1")

pyro.replace_results(bestResults, names, 'data_competing.json.template', 'data_optimized.json')