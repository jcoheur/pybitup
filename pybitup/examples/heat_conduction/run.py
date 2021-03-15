import heat_conduction
import pybitup

case_name = "heat_conduction"
input_file_name = "{}.json".format(case_name)

heat_conduction_model = {}
heat_conduction_model[case_name] = heat_conduction.HeatConduction()

# Sampling
post_dist = pybitup.solve_problem.Sampling(input_file_name)
post_dist.sample(heat_conduction_model)
post_dist.__del__()

# Propagation using PCE
post_dist = pybitup.solve_problem.Propagation(input_file_name)
post_dist.propagate(heat_conduction_model)
post_dist.__del__()

# Sensitivity analysis with kernel method
post_dist = pybitup.sensitivity_analysis.SensitivityAnalysis(input_file_name, case_name)

pybitup.post_process.post_process_data(input_file_name)
