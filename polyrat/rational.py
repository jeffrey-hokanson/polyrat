import numpy as np
import scipy.linalg
import scipy.optimize
from .basis import *
from .polynomial import *
from .skiter import *
from .rational_ratio import *
from copy import deepcopy


class RationalFunction:
	pass


class RationalApproximation:
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

class SKRationalApproximation(RationalApproximation, RationalRatio):
	r"""

	Parameters
	----------
	
	"""

	def __init__(self, num_degree, denom_degree, refine = True, norm = 2, 
		Basis = None, rebase = True, maxiter = 20, verbose = True):

		RationalApproximation.__init__(self, num_degree, denom_degree)
		self.refine = refine
		self.norm = norm
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
			self.numerator, self.denominator = skfit_rebase(
				X, y, self.num_degree, self.denom_degree,
				maxiter = self.maxiter, verbose = self.verbose, norm = self.norm
				)
		else:
			num_basis = self.Basis(X, self.num_degree)	
			denom_basis = self.Basis(X, self.denom_degree)	
			P = num_basis.basis()
			Q = denom_basis.basis()
		
			a, b = skfit(y, P, Q, maxiter = self.maxiter, verbose = self.verbose, norm = self.norm)

			self.numerator = Polynomial(num_basis, a)
			self.denominator = Polynomial(denom_basis, b)

		if self.refine:
			a, b = rational_ratio_optimize(y, self.P, self.Q, self.a, self.b, norm = self.norm)

			if self.verbose:
				res_norm = np.linalg.norm( (P @ a)/(Q @ b) - y, self.norm)
				print(f"final residual norm {res_norm:21.15e}")
				

	def __call__(self, X):
		p = self.numerator(X)
		q = self.denominator(X)
		return p/q	
					

