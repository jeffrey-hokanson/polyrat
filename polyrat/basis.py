import numpy as np
from numpy.polynomial.polynomial import polyvander, polyder, polyroots
from numpy.polynomial.legendre import legvander, legder, legroots 
from numpy.polynomial.chebyshev import chebvander, chebder, chebroots
from numpy.polynomial.hermite import hermvander, hermder, hermroots
from numpy.polynomial.laguerre import lagvander, lagder, lagroots
from functools import lru_cache
from .index import *

class PolynomialBasis:
	pass


class TensorProductPolynomialBasis(PolynomialBasis):
	def _vander(self, *args, **kwargs):
		raise NotImplementedError
	def _der(self, *args, **kwargs):
		raise NotImplementedError
	def _roots(self, *args, **kwargs):
		raise NotImplementedError

	def __init__(self, X, degree):
		try:
			self.degree = int(degree)
			self._indices = total_degree_index(X.shape[1], degree)
			self.mode = 'total'
		except (TypeError, ValueError):
			self.degree = np.copy(degree)
			self._indices = max_degree_index(self.degree)
			self.mode = 'max'
	
		self._set_scale(X)
		self.X = np.copy(X)
		self.dim = X.shape[1]

	def _set_scale(self, X):	
		self._lb = np.min(X, axis = 0)
		self._ub = np.max(X, axis = 0)

	def _scale(self, X):
		r""" Scale coordinates to [-1,1]
		"""
		return 2*(X-self._lb[None,:])/(self._ub[None,:] - self._lb[None,:]) - 1


	def vandermonde(self, X):
		r""" Construct the Vandermonde matrix
		"""
		X = self._scale(X)
		
		if self.mode is 'total':		 
			V_coordinate = [self._vander(X[:,k], self.degree) for k in range(self.dim)]
		elif self.mode is 'max':
			V_coordinate = [self._vander(X[:,k], d) for k,d in enumerate(self.degree)]
			
		
		V = np.ones((X.shape[0], len(self._indices)), dtype = X.dtype)
		for j, alpha in enumerate(self._indices):
			for k in range(self.dim):
				V[:,j] *= V_coordinate[k][:,alpha[k]]
		return V
		

	@lru_cache(maxsize = 1)
	def basis(self):
		r""" The basis for the input coordinates
		""" 
		return self.vandermonde(self.X)


class MonomialPolynomialBasis(TensorProductPolynomialBasis):
	def _vander(self, *args, **kwargs):
		return polyvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return polyder(*args, **kwargs)
	def _roots(self, *args, **kwargs):
		return polyroots(*args, **kwargs)


class LegendrePolynomialBasis(TensorProductPolynomialBasis):
	def _vander(self, *args, **kwargs):
		return legvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return legder(*args, **kwargs)
	def _roots(self, *args, **kwargs):
		return legroots(*args, **kwargs)

class ChebyshevPolynomialBasis(TensorProductPolynomialBasis):
	def _vander(self, *args, **kwargs):
		return chebvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return chebder(*args, **kwargs)
	def _roots(self, *args, **kwargs):
		return chebroots(*args, **kwargs)

class HermitePolynomialBasis(TensorProductPolynomialBasis):
	def _vander(self, *args, **kwargs):
		return hermvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return hermder(*args, **kwargs)
	def _roots(self, *args, **kwargs):
		return hermroots(*args, **kwargs)
	
	def _set_scale(self, X):
		self._mean = np.mean(X, axis = 0)
		self._std = np.std(X, axis = 0)
	
	def _scale(self, X):		
		return (X - self._mean[None,:])/self._std[None,:]/np.sqrt(2)


class LaguerrePolynomialBasis(TensorProductPolynomialBasis):
	# TODO: Change scaling to match orthogonality conditions
	def _vander(self, *args, **kwargs):
		return lagvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return lagder(*args, **kwargs)
	def _roots(self, *args, **kwargs):
		return lagroots(*args, **kwargs)

	def _set_scale(self, X):
		self._lb = np.min(X, axis = 0)
		self._a = 1./np.mean(X - self._lb[None,:], axis = 0)
		
	def _scale(self, X):
		return self._a[None,:]*(X - self._lb[None,:])
		

def _update_rule_total(indices, idx):
	diff = indices - idx
	j = np.max(np.argwhere( (np.sum(np.abs(diff), axis = 1) <= 1) & (np.min(diff, axis = 1) == -1)))
	i = int(np.argwhere(diff[j] == -1))
	return i, j	

def _update_rule_max(idx, ids):
	# Determine which column to multiply by
	diff = idx - ids
	# Here we pick the *least* recent column that is one off
	# to ensure the 
	j = np.min(np.argwhere( (np.sum(np.abs(diff), axis = 1) <= 1) & (np.min(diff, axis = 1) == -1)))
	i = int(np.argwhere(diff[j] == -1))
	return i, j	


def vandermonde_arnoldi_CGS(X, degree, weight = None, mode = None):
	r""" Multivariate Vandermode with Arnoldi using classical Gram-Schmidt with reorthogonalization

	Notes
	-----
	* The use of Classical Gram-Schmidt with reorthogonalization
	  was suggested by Yuji Nakatsukasa to improve performance.
	  This uses matrix operations rather than the vector operations
	  of modified Gram-Schmidt allowing the use of more efficient 
	  BLAS3 operations.


	Parameters
	----------
	X: np.array (M, dim)
		Input coordinates
	degree: int or list of ints
		Polynomial degree.  If an int, a total degree polynomial is constructed;
		if a list, the list must be length m and a maximum degree polynomial is
		constructed.
	weight: None or np.array (M,)
		Initial vector in the Arnoldi iteration
	mode: None or ['total', 'max']
		What type of polynomial basis to construct; only matters if dim>1.
		If None, the type of basis will be automatically detected. 
	"""
	M, dim = X.shape

	if weight is None:
		weight = np.ones(M)

	if mode is None:
		try:
			degree = int(degree)
			mode = 'total'
		except (TypeError, ValueError):
			mode = 'max'
			degree = np.copy(degree)

	if mode is 'total':
		indices = total_degree_index(dim, degree)
		update_rule = _update_rule_total 
	elif mode is 'max':
		indices = max_degree_index(degree)
		update_rule = _update_rule_max
	
	Q = np.zeros((M, len(indices)), dtype = X.dtype)
	R = np.zeros((len(indices), len(indices)), dtype = X.dtype)

	iter_indices = enumerate(indices)	

	# In the first iteration we simply orthogonalize 
	k, idx = next(iter_indices)
	q = np.array(weight, dtype = X.dtype)
	R[0,0] = np.linalg.norm(q)
	Q[:,0] = q/R[0,0]


	for k, idx in iter_indices:
		i, j = update_rule(indices, idx)
		# Form new column	
		q = X[:,i] * Q[:,j]
		
		# see Alg. 6.1 in Bjo94
		# rather than doing the explicit check for orthogonality,
		# we just go ahead and do two iterations as this is sufficient
		# for double precision accuracy 
		for it in range(2):
			s = Q[:,:k].conj().T @ q
			q -= Q[:,:k] @ s
			R[:k,k] +=s

		R[k,k] = np.linalg.norm(q)
		Q[:,k] = q/R[k,k]

	return Q, R, indices



def vandermonde_arnoldi_eval(X, R, indices, mode, weight = None):
	r"""
	"""

	X = np.array(X)
	M, m = X.shape
	W = np.zeros((X.shape[0], len(indices)), dtype = X.dtype)
	if weight is None:
		weight = np.ones(M, dtype = X.dtype)	
	
	if mode is 'total':
		update_rule = _update_rule_total 
	elif mode is 'max':
		update_rule = _update_rule_max

	iter_indices = enumerate(indices)

	# First column
	next(iter_indices)
	W[:,0] = weight/R[0,0]
	
	# Now work on the remaining columns
	for k, ids in iteridx:
		i, j = _update_vec(idx[:k], ids)
		# Form new column
		w = X[:,i] * W[:,j]

		# Perform orthogonalizations
		w -= W[:,:k] @ R[:k,k]
		# TODO: unroll this loop
		#for j in range(k):
		#	w -= R[j,k]*W[:,j]
		
		W[:,k] = w/R[k,k]

	return W

class ArnoldiPolynomialBasis(PolynomialBasis):
	r""" A polynomial basis constructed using Vandermonde with Arnoldi

	Parameters
	----------
	X: array-like (M,m)
		Input coordinates
	degree: int or list of ints
		Polynomial degree.  If an int, a total degree polynomial is constructed;
		if a list, the list must be length m and a maximum degree polynomial is
		constructed.
	"""
	def __init__(self, X, degree, weight = None):
		self.X = np.copy(np.atleast_2d(X))
		self.dim = self.X.shape[1]
		try:
			self.degree = int(degree)
			self.mode = 'total'
		except (TypeError, ValueError):
			self.degree = np.copy(degree).astype(np.int)
			self.mode = 'max'
	
		self._Q, self._R, self._indices = vandermonde_arnoldi_CGS(self.X, self.degree, weight = weight)
		
	def basis(self):
		return self._Q   

	def vandermonde(self, X, weight = None):
		return vanderonde_arnoldi_eval(X, self._R, self._indices, self.mode, weight = weight)
	
