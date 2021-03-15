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
The code can be installed as a python package using the command: 

```
python -m pip install git+https://github.com/jcoheur/pybitup    
```

Add @branch_name to install a particular branch from the git. 


## Getting started 
See examples in the [pybitup-example repository](https://github.com/jcoheur/pybitup-examples)