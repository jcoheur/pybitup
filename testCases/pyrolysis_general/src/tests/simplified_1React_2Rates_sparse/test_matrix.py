from src.pyrolysis import PyrolysisCompetitiveSparse
import matplotlib.pyplot as plt
import numpy as np

test = PyrolysisCompetitiveSparse()

test.react_reader("data_competing_verification.json")
test.param_reader("data_competing_verification.json")

A = test.generate_matrix(temperature=1000)
print(A)
drhodt = np.dot(A, [100,0,0])