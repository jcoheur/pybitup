import matplotlib.pyplot as plt
import spotpy
from src.optimizer import spotpy_setup
import pandas as pd

results=[]
folder = "."
filesCSV = ["test_20Kmin.csv"]
params = [
            spotpy.parameter.Uniform('E1',low=120E3 , high=200E3,  optguess=130000),
            spotpy.parameter.Uniform('E2',low=50E3, high=100E3, optguess=80000),
            spotpy.parameter.Uniform('A1', low=2 , high=10, optguess=0.15),
            spotpy.parameter.Uniform('A2', low=2, high=7, optguess=0.15),
]

spotpy_setup = spotpy_setup(files=filesCSV, params=params, folder=folder,
                            scheme_file="data_parallel.json",
                            pyro_type="PyrolysisParallel")
rep=200

sampler = spotpy.algorithms.sceua(spotpy_setup, dbname='SCEUA', dbformat='ram', alt_objfun="rmse_multiple_files", save_sim=False)

sampler.sample(rep,ngs=4)
names = spotpy.analyser.get_parameternames(sampler.getdata())
best = spotpy.analyser.get_best_parameterset(sampler.getdata())

# spotpy_setup.plotBest(bestResultVector=best[0])  #Doesn't work becuase 'spotpy_setup' object has no attribute 'plotBest'
bestResults = [x for x in best[0]]

resultCSV = pd.DataFrame.from_dict({"Params":names, "Value":bestResults})
resultCSV.to_csv(folder+'/'+"results_1")

plt.show()
