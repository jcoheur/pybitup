import matplotlib.pyplot as plt
import spotpy
from src.optimizer import spotpy_setup
import pandas as pd

results=[]
folder = "."
# filesCSV = [f for f in os.listdir(folder) if f.endswith('.csv')]
filesCSV = ["test_20Kmin.csv"]
# filesCSV = ["Wong_10Kmin_new.csv"]
# filesCSV = ["Wong_10Kmin_new.csv", "Wong_10Kmin_new.csv"]
params = [
            spotpy.parameter.Uniform('E1',low=25E3 , high=60E3,  optguess=47000),
            spotpy.parameter.Uniform('E2',low=100E3, high=170E3, optguess=150000),
            spotpy.parameter.Uniform('E3', low=40E3 , high=80E3, optguess=50000),
            spotpy.parameter.Uniform('E4', low=40E3, high=60E3, optguess=51000),
            spotpy.parameter.Uniform('A1', low=1, high=6, optguess=3.9),
            spotpy.parameter.Uniform('gamma1', low=0.1, high=0.2, optguess=0.99),
            spotpy.parameter.Uniform('gamma2', low=0.005, high=0.02, optguess=0.99),
]

spotpy_setup = spotpy_setup(files=filesCSV, params=params,
                            folder=folder, scheme_file="data_competing.json",
                            pyro_type="PyrolysisCompetitive")
rep=2000

sampler = spotpy.algorithms.sceua(spotpy_setup, dbname='SCEUA', dbformat='ram',alt_objfun="rmse_multiple_files")
#sampler=spotpy.algorithms.sceua(spotpy_setup, dbname='SCEUA', dbformat='csv', parallel="mpi",alt_objfun="rmse_multiple_files")
sampler.sample(rep,ngs=8)
results.append(sampler.getdata())
names = spotpy.analyser.get_parameternames(sampler.getdata())
best = spotpy.analyser.get_best_parameterset(sampler.getdata())

spotpy_setup.plotBest(bestResultVector=best[0])
bestResults = [x for x in best[0]]

resultCSV = pd.DataFrame.from_dict({"Params":names, "Value":bestResults})
resultCSV.to_csv(folder+'/'+"results_1")

plt.show()
