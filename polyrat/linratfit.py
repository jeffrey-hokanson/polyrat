""" Linearized Rational Approximation
"""
import numpy as np
from .basis import *
from . import ArnoldiPolynomialBasis
from .skiter import _minimize_2_norm

from .rational import RationalApproximation, RationalRatio
from .polynomial import Polynomial

import scipy.sparse as ss
from scipy.sparse.linalg import LinearOperator




def _linearized_2norm(P, Q, Y):
	# TODO: This should implement a LinearOperator
	pass



def linearized_ratfit(X, y, num_degree, denom_degree, Basis = ArnoldiPolynomialBasis, simultaneous = False):
	r"""Construct a rational approximation by multiplying through by the denominator.

	
	[AKL+20]_


	Parameters
	----------
	X: array-like (M, dim)
		Input coordinates to rational approximation
	y: array-like (M,...)
		Output values the rational approximation should try to take
	num_degree: int or list of ints
		degree of the numerator polynomial
	denom_degree: int or list of ints
		degree of the denominator polynomial
	Basis: :class:`.PolynomialBasis`
		basis in which to construct the numerator and denominator
	simultaneous: bool
		If true, identify the numerator and denominator coefficients by solving one linear system;
		if false, identfy the denominator coefficients first and then recover the numerator coefficients.
		The first case is, in general, more numerically stable, but can recover a zero-polynomial in the denominator.

	Returns
	-------
	numerator: :class:`.Polynomial`
		numerator polynomial 
	denominator: :class:`.Polynomial`
		denominator polynomial 

	"""
	num_basis = Basis(X, num_degree)
	denom_basis = Basis(X, denom_degree)
	P = num_basis.vandermonde_X
	Q = denom_basis.vandermonde_X	

	# diag(y) @ Q
	yQ = np.multiply(y[:,None], Q)
	

	if simultaneous:
		A = np.hstack([P, -yQ])
		x, cond = _minimize_2_norm(A)
		a = x[:P.shape[1]]
		b = x[-Q.shape[1]:]

	elif Basis == ArnoldiPolynomialBasis:
		# In AKL+19x implementation they reduce to a problem only over b
		# by using the pseudoinverse to implicitly solve for a
		# (much like in Variable Projection)

		# In this case the basis P has orthonormal columns, so we have no
		# need for the pseudoinverse
		W = P @ (P.conj().T @ yQ) - yQ
		b, cond = _minimize_2_norm(W)
		
		# and then idenify a via the pseudo-inverse
		a = P.conj().T @ (yQ @ b)
	else:
		W = P @ np.linalg.lstsq(P, yQ, rcond = None)[0] - yQ
		b, cond = _minimize_2_norm(W)
		a = np.linalg.lstsq(P, yQ @ b, rcond = None)[0]

			
	numerator = Polynomial(num_basis, a)
	denominator = Polynomial(denom_basis, b)

	return numerator, denominator


class LinearizedRationalApproximation(RationalApproximation, RationalRatio):
	r""" Construct a rational approximation using a linearized fit

	Parameters
	----------
	num_degree: int or list of ints
		degree of the numerator polynomial
	denom_degree: int or list of ints
		degree of the denominator polynomial
	**kwargs:
		Additional keyword arguments are passed to :meth:`polyrat.linearized_ratfit` 
	"""
	def __init__(self, num_degree, denom_degree, **kwargs):
		RationalApproximation.__init__(self, num_degree, denom_degree)
		self.kwargs = kwargs

	def fit(self, X, y):
		self.numerator, self.denominator = linearized_ratfit(X, y, self.num_degree, self.denom_degree, **self.kwargs)
