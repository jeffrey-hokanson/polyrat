# PolyRat: Polynomial and Rational Function Library

[![PyPI version](https://badge.fury.io/py/polyrat.svg)](https://badge.fury.io/py/polyrat)
[![CI](https://github.com/jeffrey-hokanson/polyrat/workflows/CI/badge.svg)](https://github.com/jeffrey-hokanson/polyrat/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/jeffrey-hokanson/polyrat/badge.svg?branch=master)](https://coveralls.io/github/jeffrey-hokanson/polyrat?branch=master)
[![Documentation Status](https://readthedocs.org/projects/polyrat/badge/?version=latest)](https://polyrat.readthedocs.io/en/latest/?badge=latest)


## Installation

    > pip install --upgrade polyrat


## Usage

Using PolyRat follows the general pattern of [scikit-learn](https://scikit-learn.org/stable/).
For example, to construct a rational approximation of the tangent function

```python
import numpy as np
import polyrat

x = np.linspace(-1,1, 1000).reshape(-1,1)  # Input data 🚨 must be 2-dimensional
y = np.tan(2*np.pi*x.flatten())            # Output data

num_degree, denom_degree = 10, 10          # numerator and denominator degrees 
rat = polyrat.StabilizedSKRationalApproximation(num_degree, denom_degree)
rat.fit(x, y)
```

After constructing this approximation, we can then evaluate 
the resulting approximation by calling the class-instance

```python
y_approx = rat(x)		# Evaluate the rational approximation on X
```

Comparing this to training data, we note
that this degree-(10,10) approximation is highly accurate 
<p align="center">
<img src="tan.png" alt="A rational approximation of the tangent function" height="400" style="display: block; margin: 0 auto" />
</p>




## Reproducibility

This repository contains the code to reproduce the figures in the associated papers

* [Multivariate Rational Approximation Using a Stabilized Sanathanan-Koerner Iteration](https://arxiv.org/abs/2009.10803)
  in [Reproducibility/Stabilized_SK](Reproducibility/Stabilized_SK)


## Related Projects

* [baryrat](https://github.com/c-f-h/baryrat): Pure python implementation of the AAA algorithm
* [Block-AAA](https://github.com/nla-group/block_aaa): Matlab implementation of a matrix-valued AAA variant
* [RationalApproximations](https://github.com/billmclean/RationalApproximations): Julia implementation AAA variants
* [RatRemez](https://github.com/sfilip/ratremez) Rational Remez algorithm (Silviu-Ioan Filip)
* [BarycentricDC](https://github.com/sfilip/barycentricDC) Barycentric Differential Correction (see [SISC paper](https://doi.org/10.1137/17M1132409))


