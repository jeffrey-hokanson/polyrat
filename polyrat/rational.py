import abc
import numpy as np
import scipy.linalg
import scipy.optimize
from .basis import *
from .polynomial import *
from .rational_ratio import *
from copy import deepcopy


class RationalFunction(abc.ABC):
	r"""An abstract base class for rational functions
	"""
	def __call__(self, X):
		return self.eval(X)
	
	def eval(self, X):
		r""" Evaluate the rational function at the given points

		Parameters
		----------
		X: array-like (M, dim)
			Input coordinates to evaluate the rational function
		"""
		return self.__call__(X)

class RationalApproximation(RationalFunction):
	def __init__(self, num_degree, denom_degree):
		self.num_degree = num_degree
		self.denom_degree = denom_degree
	
	@abc.abstractmethod
	def fit(self, X, y, weight = None):
		r""" Construct a rational approximation for the given data


		Parameters
		----------
		X: array-like (M, dim)
			Input coordinates to rational approximation
		y: array-like (M,...)
			Output values the rational approximation should try to take
		weight: None
			Optional weighting for those methods that support it
		"""
		raise NotImplementedError


class RationalRatio(RationalFunction):
	r"""A rational function as a ratio of two polynomials
	"""
	def __init__(self, numerator, denominator):
		self.numerator = deepcopy(numerator)
		self.denominator = deepcopy(denominator)

	@property
	def P(self):
		return self.numerator.basis.vandermonde_X

	@property
	def Q(self):
		return self.denominator.basis.vandermonde_X

	@property
	def a(self):
		return self.numerator.coef

	@property
	def b(self):
		return self.denominator.coef
	
	def eval(self, X):
		p = self.numerator(X)
		q = self.denominator(X)
		return np.multiply(1./q.reshape(-1, *[1 for i in p.shape[1:]]), p)



	def refine(self, X, y, norm = 2, verbose = False, **kwargs):
		r""" Refine the rational approximation using optimization

		The result of many algorithms does not yield a rational approximation
		that satisfies the first order necessary conditions for optimality.
		Calling this method after calling fit improves the approximation to optimality
		"""
		a, b = rational_ratio_optimize(y, self.P, self.Q, self.a, self.b, norm = norm, verbose = verbose, **kwargs)

		self.numerator.coef = a
		self.denominator.coef = b	

		#if verbose:
		#	res_norm = np.linalg.norm( (self.P @ a)/(self.Q @ b) - y, norm)
		#	print(f"final residual norm {res_norm:21.15e}")



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






			
