import os

os.chdir("ito_sde/1_param")
os.system('clean.py')
os.chdir("../../")

os.chdir("ito_sde/2_param")
os.system('clean.py')
os.chdir("../../")