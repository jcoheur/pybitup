import os


list_test_cases = [ 'heat_capacity', 
                    'heat_conduction', 
                    'pyrolysis', 
                    'pyrolysis_competitive', 
                    'pyrolysis_parallel_fran_model', 
                    'sampling', 
                    'spring_model',
                    'pyrolysis_multiple',
                    'pato_distribution']

for test_name in list_test_cases: 
    os.chdir(test_name)
    cmd1 = 'python clean.py'
    os.system(cmd1)
    os.chdir('../')