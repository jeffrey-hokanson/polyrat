import abc
import numpy as np
from .basis import *
from copy import deepcopy
import scipy.linalg
import cvxpy as cp
from .util import _zeros


class Polynomial:
	r""" Define a polynomial function

	Parameters
	----------
	basis: :class:`.Basis`
		An instantiated instance of a basis
	coef: numpy array
		Coefficients corresponding the ordered basis elements.
	"""
	def __init__(self, basis, coef):
		self.basis = deepcopy(basis)
		self.coef = np.copy(coef)

	def __call__(self, X):
		return self.eval(X)

	@property
	def degree(self):
		return self.basis.degree	

	def eval(self, X):
		#return self.basis.vandermonde(X) @ self.coef
		return np.einsum('ij,j...->i...', self.basis.vandermonde(X), self.coef)

	def derivative(self, X):
		r""" Compute the derivative 
		"""
		return np.einsum('ijk,j...->i...k', self.basis.vandermonde_derivative(X), self.coef)

	def roots(self, *args, **kwargs):
		assert self.basis.dim == 1, "Must have a single variable as input"
		assert np.prod(self.coef[0].shape) == 1, "Must have one dimensional output"
		return self.basis.roots(self.coef.reshape(-1), *args, **kwargs)	


def _polynomial_fit_least_squares(P, Y, P_orth = False):
	M, m = P.shape
	
	coef = _zeros((P.shape[1], *Y.shape[1:]), P, Y)

	if P_orth:
		for j, idx in enumerate(np.ndindex(Y.shape[1:])):
			coef[(slice(m), *idx)] = P.T.conj() @ Y[(slice(M), *idx)]
	else:
		Q, R = scipy.linalg.qr(P, mode = 'economic')
		for j, idx in enumerate(np.ndindex(Y.shape[1:])):
			a = scipy.linalg.solve_triangular(R, Q.T.conj() @ Y[(slice(M), *idx)])
			coef[(slice(m), *idx)] = a
				
	return coef

def _polynomial_fit_pnorm(P, y, norm, **kwargs):
	if np.iscomplexobj(P) or np.iscomplexobj(y):
		a = cp.Variable(P.shape[1], complex = True)
	else:
		a = cp.Variable(P.shape[1], complex = False)
	
	prob = cp.Problem(cp.Minimize(cp.norm(P @ a - y, p = norm)))
	prob.solve(**kwargs)
	return a.value

class PolynomialApproximation(Polynomial):
	def __init__(self, degree, Basis = None, norm = 2):
		assert norm >= 1
		if Basis is None:
			from .arnoldi import ArnoldiPolynomialBasis
			Basis = ArnoldiPolynomialBasis
		self.Basis = Basis
		self._degree = degree
		self.norm = norm

	@property
	def degree(self):
		return self._degree

	def fit(self, X, y, **kwargs):
		from .arnoldi import ArnoldiPolynomialBasis
		self.basis = self.Basis(X, self.degree)
		P = self.basis.vandermonde_X
		if self.norm == 2 or self.norm == 2.:
			self.coef = _polynomial_fit_least_squares(P, y, isinstance(self.Basis, ArnoldiPolynomialBasis))
		else:
			self.coef = _polynomial_fit_pnorm(P, y, self.norm, **kwargs)
		


