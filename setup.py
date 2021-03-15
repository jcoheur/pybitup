from setuptools import setup

setup(
    name='pybitup',
    version='1.0',
    packages=['pybitup'],
    url='https://github.com/jcoheur/pybitup/',
    license='',
    author='Joffrey Coheur',
    author_email='joffrey.coheur@uliege.be',
    description='Python Bayesian Inference Toolbox and Uncertainty Propagation',
	install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'seaborn',
                      'jsmin',
                      'mpi4py',
                      'sobol_seq',
                      'matplotlib',
                      'tikzplotlib']
)
