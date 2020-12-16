r""" Vandermonde with Arnoldi basis


"""
import numpy as np
from .basis import PolynomialBasis
from .index import *

def _update_rule(indices, idx):
	diff = indices - idx
	j = np.min(np.argwhere( (np.sum(np.abs(diff), axis = 1) <= 1) & (np.min(diff, axis = 1) == -1)))
	i = int(np.argwhere(diff[j] == -1))
	return i, j	

#def _update_rule_max(idx, ids):
#	# Determine which column to multiply by
#	diff = idx - ids
#	# Here we pick the *least* recent column that is one off
#	j = np.min(np.argwhere( (np.sum(np.abs(diff), axis = 1) <= 1) & (np.min(diff, axis = 1) == -1)))
#	i = int(np.argwhere(diff[j] == -1))
#	return i, j	


def vandermonde_arnoldi_CGS(X, degree, weight = None, mode = None):
	r""" Multivariate Vandermode with Arnoldi using classical Gram-Schmidt with reorthogonalization


	This function uses the Arnoldi proceedure to construct an orthogonal
	basis for polynomials.  In the univariate case, this corresponds to
	constructing a Krylov subspace for the starting vector :math:`\mathbf{b}`
	and matrix :math:`\text{diag}(\mathbf{x})`; see [BNT19x]_. The multivariate
	case uses a similar approach, but care is taken to use the right
	combination of input coordinates.; see [AKL+20]_.


	Notes
	-----
	* The use of Classical Gram-Schmidt with reorthogonalization
	  was suggested by Yuji Nakatsukasa to improve performance.
	  This uses matrix operations rather than the vector operations
	  of modified Gram-Schmidt allowing the use of more efficient 
	  BLAS3 operations.


	Parameters
	----------
	X: np.array
		Input coordinates of size (M, dim)
	degree: int or list of ints
		Polynomial degree.  If an int, a total degree polynomial is constructed;
		if a list, the list must be length m and a maximum degree polynomial is
		constructed.
	weight: None or np.array (M,)
		Initial vector in the Arnoldi iteration
	mode: None or ['total', 'max']
		What type of polynomial basis to construct; only matters if dim>1.
		If None, the type of basis will be automatically detected.

	
	Returns
	-------
	Q: :class:`~numpy:numpy.ndarray`
		Orthonormal basis for the desired polynomials on the input coordinates.
	R: :class:`~numpy:numpy.ndarray`
		Matrix containing orthogonalization information; needed to evaluate polynomial at new coordinates.
	indices: :class:`~numpy:numpy.ndarray`
		List of indices showing the order in which the basis was constructed.	 
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

	if mode == 'total':
		indices = total_degree_index(dim, degree)
	elif mode == 'max':
		indices = max_degree_index(degree)
	
	Q = np.zeros((M, len(indices)), dtype = X.dtype)
	R = np.zeros((len(indices), len(indices)), dtype = X.dtype)

	iter_indices = enumerate(indices)	

	# In the first iteration we simply orthogonalize 
	k, idx = next(iter_indices)
	q = np.array(weight, dtype = X.dtype)
	R[0,0] = np.linalg.norm(q)
	Q[:,0] = q/R[0,0]


	for k, idx in iter_indices:
		i, j = _update_rule(indices, idx)
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



def vandermonde_arnoldi_eval(X, R, indices, weight = None):
	r""" Evaluate a Vandermonde with Arnoldi polynomial

	Parameters
	----------
	X: array_like
		Coordiantes at which to evaluate the polynomial 
	R: :class:`~numpy:numpy.ndarray`
		Upper triangular matrix containing orthogonalization information from 
		:class:`~polyrat:polyrat.vandermonde_arnoldi_CGS`.
	indices: :class:`~numpy:numpy.ndarray`
		The ordering of polynomial powers returned from
		:class:`~polyrat:polyrat.vandermonde_arnoldi_CGS`.
	weight: None or array_like
		Starting vector for Arnoldi; by default is the ones vector.
		In general, this should not change without good reason.

	Returns
	-------
	W: :class:`~numpy:numpy.ndarray`
		Polynomial basis evaluated at the specified points.
	"""

	X = np.array(X)
	M, m = X.shape
	W = np.zeros((X.shape[0], len(indices)), dtype = X.dtype)
	if weight is None:
		weight = np.ones(M, dtype = X.dtype)	
	

	iter_indices = enumerate(indices)

	# First column
	next(iter_indices)
	W[:,0] = weight/R[0,0]
	
	# Now work on the remaining columns
	for k, ids in iter_indices:
		i, j = _update_rule(indices[:k], ids)
		# Form new column
		w = X[:,i] * W[:,j]

		# Perform orthogonalizations
		w -= W[:,:k] @ R[:k,k]
		
		W[:,k] = w/R[k,k]

	return W


def vandermonde_arnoldi_eval_der(X, R, indices, weight = None, V = None):
	r""" Evaluate the derivative of Vandermonde with Arnoldi polynomial basis

	Parameters
	----------
	X: array_like
		Coordiantes at which to evaluate the polynomial 
	R: :class:`~numpy:numpy.ndarray`
		Upper triangular matrix containing orthogonalization information from 
		:class:`~polyrat:polyrat.vandermonde_arnoldi_CGS`.
	indices: :class:`~numpy:numpy.ndarray`
		The ordering of polynomial powers returned from
		:class:`~polyrat:polyrat.vandermonde_arnoldi_CGS`.
	weight: None or array_like
		Starting vector for Arnoldi; by default is the ones vector.
		In general, this should not change without good reason.
	V: :class:`~numpy:numpy.ndarray`
		Evaluation of the polynomial without the derivative at X.
	
	Returns
	-------
	W: :class:`~numpy:numpy.ndarray`
		Polynomial basis deriative evaluated at the specified points.
	"""
	if V is None:
		V = vandermonde_arnoldi_eval(X, R, indices, weight = weight)
		
	M = X.shape[0]
	N = R.shape[1]
	n = X.shape[1]
	DV = np.zeros((M, N, n), dtype = (R[0,0] * X[0,0]).dtype)

	for ell in range(n):
		index_iterator = enumerate(indices)
		next(index_iterator)
		for k, ids in index_iterator:
			i, j = _update_rule(indices[:k], ids)
			# Q[:,k] = X[:,i] * Q[:,j] - sum_s Q[:,s] * R[s, k]
			if i == ell:
				DV[:,k,ell] = V[:,j] + X[:,i] * DV[:,j,ell] - DV[:,0:k,ell] @ R[0:k,k] 
			else:
				DV[:,k,ell] = X[:,i] * DV[:,j,ell] - DV[:,0:k,ell] @ R[0:k,k] 
			DV[:,k,ell] /= R[k,k]	
	
	return DV


class ArnoldiPolynomialBasis(PolynomialBasis):
	r""" A polynomial basis constructed using Vandermonde with Arnoldi.

	A polynomial basis constructed with Vandermonde with Arnoldi;
	see [BNT19x]_, [AKL+20]_ for discussion.

	Notes
	-----
	* In order to compute the roots in the univariate case we convert to Legendre polynomial. 

	"""
	def __init__(self, X, degree, weight = None):
		PolynomialBasis.__init__(self, X, degree)
		self._Q, self._R, self._indices = vandermonde_arnoldi_CGS(self.X, self.degree, weight = weight)
		self._Q.flags.writeable = False	
	
	@property
	def vandermonde_X(self):
		return self._Q   

	def vandermonde(self, X, weight = None):
		return vandermonde_arnoldi_eval(X, self._R, self._indices, weight = weight)

	def vandermonde_derivative(self, X, weight = None):
		if np.array_equal(X, self.X):
			return vandermonde_arnoldi_eval_der(X, self._R, self._indices,  weight = weight, V = self._Q)
		else:
			return vandermonde_arnoldi_eval_der(X, self._R, self._indices,  weight = weight)

	def roots(self, coef, *args, **kwargs):
		from .basis import LegendrePolynomialBasis
		from .polynomial import PolynomialApproximation
		y = self.vandermonde_X @ coef
		poly = PolynomialApproximation(self.degree, Basis = LegendrePolynomialBasis)
		poly.fit(self.X, y)
		roots = poly.roots()
		
		return roots

