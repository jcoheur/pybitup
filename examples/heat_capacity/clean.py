import os
import pathlib
import shutil
import glob


# Remove the output folder
path_output = pathlib.Path('output')
if path_output.exists():
    shutil.rmtree('output')

# Get all temporary (tmp_) filenames
tmp_files = glob.glob('tmp_*')

# Remove all tmp_files 
for tmp_filename in tmp_files: 
    os.remove(tmp_filename)

# Remove all tex files 
tmp_files = glob.glob('*.tex')

# Remove all tmp_files 
for tmp_filename in tmp_files: 
    os.remove(tmp_filename)

# Remove covariance matrix
try: os.remove("cov.csv")
except: pass
