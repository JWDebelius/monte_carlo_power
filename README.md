[![Build Status](https://travis-ci.org/jwdebelius/Machiavellian.svg?branch=master)](https://travis-ci.org/jwdebelius/Machiavellian)


# Emperical Power
Simulations and benchmarking for Debelius *et al*. Simulation notebooks are designed to run in a series: simulating data, calculating emperical power, and then calculating power based on a fit effect size. The analysis notebook demonstrates the application of the method.

## Installation
The recommended way to install this repository is using the conda pacakge manager through [miniconda](http://conda.pydata.org/miniconda.html).

Dependencies can be installed using the [conda_environment.txt](conda_enviroment.txt) and [pip_requirements.txt](pip_requirements.txt) file.

```bash
$ conda create --name power --file conda_enviroment.txt
$ source activate power
$ pip install -r requirements.txt
$ pip install git+https://github.com/jwdebelius/monte_carlo_power --no-deps
```

## Simulation Notebooks

The method is validated and benchmarked using simulated data. The method has been analyzed using four common types of parametric data, and two permutative tests for distance matrices.

 All simulated data can be downloaded from **[FTP]** address. This includes all simulated data, emperically calculated power, and emperical summary files. 

The simulated data should be place in the `ipynb` directory. From the parent repository directory, it can be downloaded as

```bash
wget [ftp]
tar -czf simulations.tgz
```

[**1-Build Simulations**](ipynb/1-Build%20Simulations.ipynb): builds simulated data parametric tests and distance matrices for permuatitive tests

[**2-Power for Parametric Distributions**](ipynb/2-Power%20for%20Parametric%20Distributions.ipynb): calculates emperical power and distribution-based power for parametric distributions

[**3-Power for distance permutations**](ipynb/3-Power%20for%20distance%20permutations.ipynb): calculates power for emperical for permutative tests

[**4-Comparisons of Power Calculations**](ipynb/4-Comparisons%20of%20Power%20Calculations.ipynb): fits the emperical power curves, and compares the performance of the distribution-based method, emperical method, and fit method

## Credits

[`power.py`](https://github.com/biocore/scikit-bio/blob/master/skbio/stats/power.py) and [`test_power.py`](https://github.com/biocore/scikit-bio/blob/master/skbio/stats/tests/test_power.py) are modified from [scikit-bio](www.scikit-bio.org) 0.5.0. The code is relased under a BSD-2.0 license; copyright (c) 2013 scikit-bio development team.

These were written and modified by Justine Debelius ([@jwdebelius](https://github.com/jwdebelius)), Greg Caporaso ([@gregcaporaso](https://github.com/gregcaporaso)), Jai Ram Rideout ([@jairideout](https://github.com/jairideout)), Evan Bolyen ([@eboylen](https://github.com/ebolyen)), and Vivek Rai ([@vivekitkgp](https://github.com/vivekiitkgp)).
