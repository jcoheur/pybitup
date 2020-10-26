# pybitup
Python Bayesian Inference Toolbox and Uncertainty Propagation 

## Features 

1. Bayesian calibration for parameter inference
    1. Experimental data or synthetic data, with one or several physical models. 
    2. Markov chain Monte Carlo (MCMC) method using Metropolis-Hastings algorithms, Ito Stochastic differential equation. 
    3. Compute correlation among input variables.  
    4. Posterior predictive checks. 
    
2. Uncertainty propagation
    1. Monte carlo or polynomial chaos methods. 
    2. Using *labelled* distributions or propagate MCMC samples with distributions that can be correlated. 
    
3. Sensitivity analysis
    1. From MCMC chains. 
    2. Monte Carlo method or Kernel method. 
    
    
## To Do 

1. Sensitivity analysis module still needs to be developped. 
2. Couple PCE surrogate model to perform sensitivity analysis. 


## Installation 
from your terminal/powershell run
python setup.py install 


#### Manual installation on Windows : 

* Copy-past the folder in the site packages: 
1. Create a pybitup directory. Example: in the site-packages folder of python 
```
C:\Users\USER\AppData\Local\Programs\Python\Python37\Lib\site-packages
```
2. Copy-paste the following objects: 
```
__init__
metropolis_hastings_algorithms
post_process
inference_problem
bayesian_inference
distributions
sample_dist
```


* Change the environment variable related to the python path 

1. Add to you pythonpath the path to pybitup directory 
(e.g. in French) 
```
Propriétés système > Variables d'environnement > PYTHONPATH 
```
