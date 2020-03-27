import src.pyrolysis as pyro
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results_1')
param_names = list(df.Params)
values = list(df.Value)

results = pyro.replace_results(values, param_names, 'data_competing.json.template', 'data_optimized.json')
