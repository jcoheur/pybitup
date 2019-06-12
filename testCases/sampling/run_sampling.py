import pybit
import json
from jsmin import jsmin
import numpy as np 

#input_file_name = "sampling_gaussian.json"
input_file_name = "sampling_multivariate_gaussian.json"

my_dist = pybit.sample_dist.SolveProblem(input_file_name)
my_dist.sample()
my_dist.post_process_dist()


