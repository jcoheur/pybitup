import pathlib 
from setuptools import setup
import versioneer

path_to_requirement_file = pathlib.Path("requirements.txt")
with open(path_to_requirement_file) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]
                    
setup(
    name='pybitup',
	version=versioneer.get_version(),
	cmdclass=versioneer.get_cmdclass(),
    packages=['pybitup', 'pybitup.pce'],
    url='https://github.com/jcoheur/pybitup/',
    license='',
    author='Joffrey Coheur',
    author_email='joffrey.coheur@uliege.be',
    description='Python Bayesian Inference Toolbox and Uncertainty Propagation',
	long_description=readme,
	install_requires=requirements
)
