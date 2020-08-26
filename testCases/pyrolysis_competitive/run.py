# Packages for the pyrolysis model
import competitive_reaction

# Packages for stochastic inference
import pybitup

case_name = "pyrolysis_competitive"
input_file_name = "{}.json".format(case_name) 

pyrolysis_model = {}
pyrolysis_model["case_name"] = competitive_reaction.SetCompetitiveReaction()

post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(pyrolysis_model)
post_dist.__del__()

pybitup.post_process.post_process_data(input_file_name)


