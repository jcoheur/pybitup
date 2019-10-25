from setuptools import setup

setup(
    name='pyBIT',
    version='1.0',
    packages=['pybit'],
    url='https://github.com/jcoheur/pybitup/',
    license='',
    author='Joffrey Coheur',
    author_email='',
    description='Python Bayesian Inference Toolbox and Uncertainty Propagation',
	install_requires=['matplotlib', 'numpy','tikzplotlib']
)
