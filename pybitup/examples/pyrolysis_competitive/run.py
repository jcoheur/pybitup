# Packages for the pyrolysis model
import competitive_reaction

# Packages for stochastic inference
import pybitup

input_file_name = "pyrolysis_competitive.json"

pyrolysis_model = {}
pyrolysis_model["pyrolysis_competitive_366_K_per_min"] = competitive_reaction.SetCompetitiveReaction()
pyrolysis_model["pyrolysis_competitive_10_K_per_min"] = competitive_reaction.SetCompetitiveReaction()


# Sampling 
post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(pyrolysis_model)
post_dist.__del__()

# Post Process
pybitup.post_process.post_process_data(input_file_name)


