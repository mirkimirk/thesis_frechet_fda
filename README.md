# Fréchet Regression and Functional Data

[![image](https://img.shields.io/github/actions/workflow/status/mirkimirk/frechet_fda/main.yml?branch=main)](https://github.com/mirkimirk/frechet_fda/actions?query=branch%3Amain)
[![image](https://codecov.io/gh/mirkimirk/frechet_fda/branch/main/graph/badge.svg)](https://codecov.io/gh/mirkimirk/frechet_fda)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/mirkimirk/frechet_fda/main.svg)](https://results.pre-commit.ci/latest/github/mirkimirk/frechet_fda/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## About this project

This repository contains my master's thesis for the M.Sc. Economics programme at
University Bonn. It contains both the code, as well as the tex file for creating the
paper. This thesis is mostly based on the papers "Functional Data Analysis for Density Functions by
Transformation to a Hilbert Space" (2016) and "Fréchet Regression for Random Objects with
Euclidean Predictors" (2019) from Alexander Petersen and Hans-Georg Müller.

### What to look for
The three Jupyter notebooks in src / frechet_fda contain all the code
done to illustrate the methods and compute the simulations. They are somewhat differentiated
into a notebook that tries to reproduce methods and results from Petersen & Müller (2016),
one to reproduce the methods from Petersen & Müller (2019) as well as do a simulation
study, and one to produce and save all the plots to use in the thesis.

### Important helper files
The file function_class.py contains the ubiquitous Function object I defined to represent
the distribution objects in my code. The file function_tools.py contains most of the
heavy used tools (including the LQD and inverse LQD transformation).

## Getting started

To get started, create and activate the environment with

```console
$ conda env create -f environment.yml
$ conda activate frechet_fda
```

Using this environment, it should be possible to just run the code in the Jupyter
notebooks.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
