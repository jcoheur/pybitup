# Packages for the pyrolysis model
from pyrolysis_general.src.pyrolysis import PyrolysisCompetitive 
from pyrolysis_general.src.read_experiments import ReadExperiments
import competitive_reaction

# Packages for stochastic inference
import pybitup

case_name = "pyrolysis_competitive"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = competitive_reaction.SetCompetitiveReaction()

post_dist = pybitup.sample_dist.SolveProblem(input_file_name)
post_dist.sample(pyrolysis_model)
post_dist.post_process_dist()


