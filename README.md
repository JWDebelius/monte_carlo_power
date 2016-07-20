# Monte Carlo Emperical Power
Simulations and benchmarking for monte carlo power.

## Installation
The recommended way to install this repository is using the conda pacakge manager through [miniconda]().

Conda dependencies can be installed in the conda enviroment, *power*

```bash
conda create --name power --file conda_enviroment.txt
source activate power
pip install git+https://github.com/jwdebelius/monte_carlo_power
```

## Credits
[`power.py`](https://github.com/biocore/scikit-bio/blob/master/skbio/stats/power.py) and [`test_power.py`](https://github.com/biocore/scikit-bio/blob/master/skbio/stats/tests/test_power.py) are modified from [scikit-bio](www.scikit-bio.org) 0.5.0. The code is relased under a BSD-2.0 license; copyright (c) 2013 scikit-bio development team.

These were written and modified by Justine Debelius ([@jwdebelius](https://github.com/jwdebelius)), Greg Caporaso ([@gregcaporaso](https://github.com/gregcaporaso)), Jai Ram Rideout ([@jairideout](https://github.com/jairideout)), Evan Bolyen ([@eboylen](https://github.com/ebolyen)), and Vivek Rai ([@vivekitkgp](https://github.com/vivekiitkgp)).