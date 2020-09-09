import numpy as np
from .basis import *
from copy import deepcopy
import scipy.linalg

class Polynomial:
	def __init__(self, basis, coef):
		self.basis = deepcopy(basis)
		self.coef = np.copy(coef)

	def __call__(self, X):
		return self.basis.vandermonde(X) @ self.coef

	def eval(self, X):
		return self.basis.vandermonde(X) @ self.coef

	def roots(self, *args, **kwargs): 
		return self.basis.roots(self.coef, *args, **kwargs)	


def _polynomial_fit_least_squares(P, y):
	coef, _, _, _ = scipy.linalg.lstsq(P, y)
	return coef.flatten()	

class PolynomialApproximation(Polynomial):
	def __init__(self, degree, Basis = None, norm = 2):
		if Basis is None:
			from .arnoldi import ArnoldiPolynomialBasis
			Basis = ArnoldiPolynomialBasis
		self.Basis = Basis
		self.degree = degree
		self.norm = norm

	def fit(self, X, y):
		self.basis = self.Basis(X, self.degree)
		P = self.basis.basis()
		if self.norm == 2:
			self.coef = _polynomial_fit_least_squares(P, y)
		else:
			raise NotImplementedError
		


