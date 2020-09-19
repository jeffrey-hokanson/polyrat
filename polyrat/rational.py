import numpy as np
import scipy.linalg
import scipy.optimize
from .basis import *
from .polynomial import *
from .skiter import *
from .rational_ratio import *
from copy import deepcopy


class RationalFunction:
	def __call__(self, X):
		return self.eval(X)
	
	def eval(self, X):
		return self.__call__(X)

class RationalApproximation(RationalFunction):
	def __init__(self, num_degree, denom_degree):
		self.num_degree = num_degree
		self.denom_degree = denom_degree


class RationalRatio(RationalFunction):
	def __init__(self, numerator, denominator):
		self.numerator = deepcopy(numerator)
		self.denominator = deepcopy(denominator)

	@property
	def P(self):
		return self.numerator.basis.basis()

	@property
	def Q(self):
		return self.denominator.basis.basis()

	@property
	def a(self):
		return self.numerator.coef

	@property
	def b(self):
		return self.denominator.coef
	
	def eval(self, X):
		p = self.numerator(X)
		q = self.denominator(X)
		return p/q	



	def refine(self, X, y, norm = 2, verbose = False, **kwargs):
		a, b = rational_ratio_optimize(y, self.P, self.Q, self.a, self.b, norm = norm, verbose = verbose, **kwargs)

		self.numerator.coef = a
		self.denominator.coef = b	

		#if verbose:
		#	res_norm = np.linalg.norm( (self.P @ a)/(self.Q @ b) - y, norm)
		#	print(f"final residual norm {res_norm:21.15e}")


class LinearizedRationalApproximation(RationalApproximation, RationalRatio):
	def __init__(self, num_degree, denom_degree, **kwargs):
		RationalApproximation.__init__(self, num_degree, denom_degree)
		self.kwargs = kwargs

	def fit(self, X, y):
		self.numerator, self.denominator = linearized_ratfit(X, y, self.num_degree, self.denom_degree, **self.kwargs)

class SKRationalApproximation(RationalApproximation, RationalRatio):
	r"""

	Parameters
	----------
	
	"""

	def __init__(self, num_degree, denom_degree, refine = False, norm = 2, 
		Basis = None, rebase = True, maxiter = 20, verbose = True, xtol = 1e-7):

		RationalApproximation.__init__(self, num_degree, denom_degree)
		self._refine = refine
		self.norm = norm
		self.xtol = float(xtol)
		#if self.norm != 2:
		#	raise NotImplementedError

		self.maxiter = int(maxiter)
		self.verbose = verbose
		self.rebase = rebase
		if Basis is None:
			Basis = LegendrePolynomialBasis
		
		self.Basis = Basis

		self.numerator = None
		self.denominator = None

	def fit(self, X, y, denom0 = None):

		X = np.array(X)
		y = np.array(y)
		assert X.shape[0] == y.shape[0], "X and y do not have the same number of rows"

		if self.rebase:
			self.numerator, self.denominator, self.hist = skfit_rebase(
				X, y, self.num_degree, self.denom_degree,
				maxiter = self.maxiter, verbose = self.verbose, norm = self.norm,
				history = True, xtol = self.xtol, denom0 = denom0,
				)
		else:
			num_basis = self.Basis(X, self.num_degree)	
			denom_basis = self.Basis(X, self.denom_degree)	
			P = num_basis.basis()
			Q = denom_basis.basis()
		
			a, b, self.hist = skfit(y, P, Q, maxiter = self.maxiter, verbose = self.verbose, norm = self.norm, history = True, 
				xtol = self.xtol, denom0 = denom0)

			self.numerator = Polynomial(num_basis, a)
			self.denominator = Polynomial(denom_basis, b)
		
		if self._refine:
			self.refine(X, y, norm = self.norm)




class RationalBarycentric(RationalFunction):
	r"""
	"""
	def __init__(self, degree):
		self.degree = int(degree)
		assert self.degree >= 0, "Degree must be non-negative"

	@property
	def num_degree(self):
		return self.degree

	@property
	def denom_degree(self):
		return self.degree






			
