# FireL1

Script to optimize fire-arrival time estimation using L<sup>1</sup> mimimization

Developed on Windows 10 using Python 2.7.x
by Lauren Hearn, February 2018
***
## 1D:

#### Dependencies:
numpy, matplotlib, pyomo

A note on Pyomo:
Before running, install pyomo via `conda install -c conda-forge pyomo`
and install all dependencies of pyomo with `conda install -c conda-forge pyomo.extras`

From the Pyomo website:
> Pyomo does not include any stand-alone optimization solvers. Consequently, most users will need to install third-party solvers to analyze optimization models built with Pyomo.
Note that Pyomo can remote launch optimization solvers on NEOS.  However, this requires the installation of the pyomo "extras".

We use the **GLPK** solver, here, which can be installed via `conda install glpk`

## 2D:

#### Dependencies:
(All of the above, plus:)

Solver: **ipopt**

Ipopt is an opensource, nonlinear solver that uses the interior point method. Info on installing can be found [here](https://www.coin-or.org/Ipopt/documentation/).

Or install via conda (**better**) using `conda install -q -y --channel cachemeorg ipopt_bin` (Note: conda uses an older version of ipopt)
#### TODOs:
- [x] get ipopt working to test
- [ ] make artificial data generation more robust
- [x] add plotting
> - [ ] plot as surface vs. 3d scatter 
