r""" Basic polynomial classes

These classes use the polynomial representations included in numpy

"""
import abc
from functools import lru_cache

try:
	from funtools import cached_property
except ImportError:
	from backports.cached_property import cached_property

import numpy as np
from numpy.polynomial.polynomial import polyvander, polyder, polyroots
from numpy.polynomial.legendre import legvander, legder, legroots 
from numpy.polynomial.chebyshev import chebvander, chebder, chebroots
from numpy.polynomial.hermite import hermvander, hermder, hermroots
from numpy.polynomial.laguerre import lagvander, lagder, lagroots

from .index import *

class PolynomialBasis(abc.ABC):
	r"""An abstract base class for polynomial bases.
	"""

	def __init__(self, X, degree):
		r"""
		
		Parameters
		----------
		X: array_like
			Array of dimensions (M, dim) specifying discrete points on which the polynomial basis will be evaluated.
		degree: :obj:`int` or :obj:`list` of :obj:`int`
			The degree of the polynomial:

			* if an int, this specifies a total degree polynomial;
			* if a list of int, this specifies a maximum degree polynomial (must match the number of columns in `X`)

		"""
		self._X = np.copy(X)
		self._X.flags.writeable = False
		try:
			self._degree = int(degree)
			self._indices = total_degree_index(self.dim, degree)
			self._mode = 'total'
		except (TypeError, ValueError):
			self._degree = np.copy(degree).astype(np.int)
			self._degree.flags.writeable = False
			assert len(self.degree) == self.dim, "maximum degree does not match the input dimension"
			self._indices = max_degree_index(self.degree)
			self._mode = 'max'

	def __str__(self):
		out =  f"<{self.__class__.__name__} of {self.mode} degree-"
		if self.mode == 'total':
			out += f"{self.degree:d}"
		elif self.mode == 'max':
			out += "("+" ".join([f"{d:d}" for d in self.degree]) + ")"
		out += f" on {self.dim:d} dimensions>"
		return out
			 

	@property
	def mode(self):
		r"""What type of polynomial basis is used"""
		return self._mode

	@property
	def X(self):
		r""" The points on which the basis was defined."""
		return self._X

	@property
	def degree(self):
		r"""The degree of the polynomial."""
		return self._degree

	@property
	def dim(self):
		r"""The dimension of the input space."""
		return self._X.shape[1]

	@property
	@abc.abstractmethod
	def vandermonde_X(self):
		r""" Alias for vandermonde(X) where X is the points where the basis was initialized

		Returns
		-------
		V: :class:`~numpy:numpy.ndarray`
			generalized Vandermonde matrix of dimensions (M, N) evaluated using the points the basis was initialized with
		"""
		pass	
		
	@abc.abstractmethod
	def vandermonde(self, X):
		r""" Construct the generalized Vandermonde matrix associated with the polynomial basis.

		For a given input :code:`X` whose rows are the vectors :math:`\mathbf x_i`,
		construct the generalized Vandermonde matrix whose entries are

		.. math::
			[\mathbf V]_{i,j} = \phi_j(\mathbf x_i).

		Parameters
		----------
		X: array_like
			Coordinates at which to evaluate the basis; of dimensions (M, dim).

		Returns
		-------
		V: :class:`~numpy:numpy.ndarray`
			generalized Vandermonde matrix; of dimensions (M, N) 
		"""
		pass

	@abc.abstractmethod
	def vandermonde_derivative(self, X):
		r""" Construct the derivative of the generalized Vandermonde matrix.

		For a given input :code:`X` whose rows are the vectors :math:`\mathbf x_i`,
		construct the 3-tensor whose entires correspond to the derivative 
		of the basis polynomial :math:`\phi_j` with respect to each coordiante 
		evaluated at :math:`\mathbf x_i`.

		.. math::
			[\mathbf V]_{i,j,k} = \left. \frac{\partial}{\partial x_k} \phi_j(\mathbf x) \right|_{\mathbf x = \mathbf x_i}.

		Parameters
		----------
		X: array-like (M, dim)
			Coordinates at which to evaluate the basis; of dimensions (M, dim).

		Returns
		-------
		DV: :class:`~numpy:numpy.ndarray`
			slice-wise derivative of generalized Vandermonde matrix; of dimensions (M, N, dim) 
		"""
		pass

	@abc.abstractmethod
	def roots(self, coef):
		r""" Compute the roots of a univariate, scalar valued polynomial.

		Given coefficients :math:`c_j` that define a polynomial

		.. math::
			p(x) = \sum_{j=1}^N c_j \phi_j(x) 

		compute the roots :math:`r_k` such that :math:`p(r_k) = 0`. 

		Parameters
		----------
		coef: array-like, (N,)
			Coefficients :math:`c_j`
		"""
		pass

class TensorProductPolynomialBasis(PolynomialBasis):
	r"""Abstract base class for polynomial bases constructed from univariate bases
	"""
	def __init__(self, X, degree):
		PolynomialBasis.__init__(self, X, degree)
		self._set_scale()
	
	@abc.abstractmethod
	def _vander(self, X, degree):
		r""" Construct a univariate Vandermonde matrix.
		
		When subclassing, this function should implement
		the same API as functions like :func:`~numpy:numpy.polynomial.polynomial.polyvander`. 

		Parameters
		----------
		X: array_like
			A one-dimensional array of coordinates
		degree: int
			Nonnegative integer for the degree

		Returns
		-------
		V: :class:`~numpy:numpy.ndarray`
			Array of shape (len(X), degree+1).
		"""
		pass


	@abc.abstractmethod
	def _der(self, coef):
		r""" Given univariate polynomial coefficients, yield the coefficients of the deriative polynomial.

		When subclassing, this function should implement
		the same API as functions like :func:`~numpy:numpy.polynomial.polynomial.polyder`. 
		
		Parameters
		----------
		coef: array_like
			Polynomial coefficients

		Returns
		-------
		der: :class:`~numpy:numpy.ndarray`
			Coefficients of deriative polynomial
		"""
		pass

	@abc.abstractmethod
	def roots(self, coef):
		r""" Given univariate polynomial coefficients, compute the roots in this basis.

		When subclassing, this function should implement
		the same API as functions like :func:`~numpy:numpy.polynomial.polynomial.polyroots`. 
		
		Parameters
		----------
		coef: array_like
			Polynomial coefficients

		Returns
		-------
		roots: :class:`~numpy:numpy.ndarray`
			Roots of the polynomial


		Notes
		-----
		Although computing the roots is more a property of a polynomial, 
		and not a basis, how we compute these roots depends strongly on the choice of basis.
		"""
		pass


	def _set_scale(self):
		r"""Perform any precomputation needed to scale the basis.

		To better condition the polynomial, we perform an affine transformation on the 
		input data; i.e., 

		.. math::
			\hat{\mathbf{x}}_i = \text{diag}(\mathbf a) \mathbf x_i + \mathbf b.

		By default we scale the basis such that :math:`\hat{\mathbf{x}}_i \in [-1,1]^d`.
		However other bases may implement a different scaling.
		"""
		self._lb = np.min(self.X, axis = 0)
		self._ub = np.max(self.X, axis = 0)

	def _scale(self, X):
		r""" Scale coordinates using an affine transform

		Parameters
		----------
		X: :class:`~numpy:numpy.ndarray`
			Coordinates to transform
		
		Returns
		-------
		Xhat: :class:`~numpy:numpy.ndarray`
			Transformed coordinates
		"""
		return 2*(X-self._lb[None,:])/(self._ub[None,:] - self._lb[None,:]) - 1

	def _inv_scale(self, X):
		r""" Perform the inverse of the affine scaling

		Parameters
		----------
		Xhat: :class:`~numpy:numpy.ndarray`
			Transformed coordinates
		
		Returns
		-------
		X: :class:`~numpy:numpy.ndarray`
			Coordinates in original space

		"""
		return X*(self._ub[None,:] - self._lb[None,:])/2.0 + (self._ub[None,:] + self._lb[None,:])/2.0

	def vandermonde(self, X):
		X = self._scale(X)
		
		if self.mode == 'total':		 
			V_coordinate = [self._vander(X[:,k], self.degree) for k in range(self.dim)]
		elif self.mode == 'max':
			V_coordinate = [self._vander(X[:,k], d) for k,d in enumerate(self.degree)]
			
		
		V = np.ones((X.shape[0], len(self._indices)), dtype = X.dtype)
		for j, alpha in enumerate(self._indices):
			for k in range(self.dim):
				V[:,j] *= V_coordinate[k][:,alpha[k]]
		return V
	
	def vandermonde_derivative(self, X):
		M, dim = X.shape
		N = len(self._indices)
		DV = np.ones((M, N, dim), dtype = X.dtype)

		Y = self._scale(X) 

		if self.mode == 'total':		 
			V_coordinate = [self._vander(Y[:,k], self.degree) for k in range(self.dim)]
		elif self.mode == 'max':
			V_coordinate = [self._vander(Y[:,k], d) for k,d in enumerate(self.degree)]

		for k in range(dim):
			for j, alpha in enumerate(self._indices):
				for q in range(self.dim):
					if q == k:
						if self.mode == 'total':
							DV[:,j,k] *= V_coordinate[q][:,0:-1] @ self._Dmat[alpha[q],:] 
						elif self.mode == 'max':
							DV[:,j,k] *= V_coordinate[q][:,0:-1] @ self._Dmat[alpha[q],:self.degree[q]]
					else:
						DV[:,j,k] *= V_coordinate[q][:,alpha[q]]

			DV[:,:,k] *= self._dscale[k] 			

		return DV	
		  
	
	@cached_property
	def vandermonde_X(self):
		V = self.vandermonde(self.X)
		V.flags.writeable = False
		return V
	
	@cached_property
	def _Dmat(self):
		r""" The matrix specifying the action of the derivative operator in this basis
		"""
		if self.mode == 'total':
			max_degree = self.degree
		elif self.mode == 'max':
			max_degree = max(self.degree)
		Dmat = np.zeros( (max_degree+1, max_degree))
		I = np.eye(max_degree + 1)
		for j in range(max_degree + 1):
			Dmat[j,:] = self._der(I[:,j])
		return Dmat


	@cached_property
	def _dscale(self):
		r""" Derivative of the scaling applied to the coordinates
		"""
		# As we assume the transformation is linear, we simply compute the finite difference
		# with a unit step size
		XX = np.zeros((2, self.dim))
		XX[1,:] = 1
		sXX = self._scale(XX)
		dscale = sXX[1] - sXX[0]
		return dscale

class MonomialPolynomialBasis(TensorProductPolynomialBasis):
	r""" A tensor product polynomial bases constructed from monomials

	Univariate monomials take the form of

	.. math::

		\phi_0(x) &= 1 \\
		\phi_1(x) &= x \\
		\phi_2(x) &= x^2 \\
		\phi_3(x) &= x^3 \\
		\vdots & \phantom{=}\vdots \\
		\phi_k(x) &= x^k 

	To improve conditioning we scale the input to the :math:`[-1,1]^d` hypercube. 

	"""
	def _vander(self, *args, **kwargs):
		return polyvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return polyder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(polyroots(*args, **kwargs))
	

class PositiveMonomialPolynomialBasis(MonomialPolynomialBasis):
	r"""A monomial scaled so the Vandermonde matrix has nonnegative entries

	This is identical to the standard MonomialPolynomialBasis except
	instead of being scaled to [-1,1], values are scaled to [0,1].
	This yields a vandermonde_X with entries in [0,1]
	"""
	def _scale(self, X):
		return (X-self._lb[None,:])/(self._ub[None,:] - self._lb[None,:]) 

	def _inv_scale(self, X):
		return X*(self._ub[None,:] - self._lb[None,:]) + self._lb[None,:]


class LegendrePolynomialBasis(TensorProductPolynomialBasis):
	r""" A tensor product polynomial basis constructed from Legendre polynomials

	`Legendre polynomials`_ are an orthogonal basis on the interval :math:`[-1,1]`:

	.. math::
		\phi_0(x) &= 1 \\ 
		\phi_1(x) &= x \\
		\phi_2(x) &= \frac12 (3x^2 -1) \\
		\phi_3(x) &= \frac12 (5x^3 -3x) \\ 
		\vdots & \phantom{=}\vdots \\
		\phi_k(x) &= \frac{1}{2^k k!} \frac{\partial^k}{\partial x^k} (x^2 - 1)^k 

	.. _Legendre polynomials: https://en.wikipedia.org/wiki/Legendre_polynomials
	"""

	def _vander(self, *args, **kwargs):
		return legvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return legder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(legroots(*args, **kwargs))

class ChebyshevPolynomialBasis(TensorProductPolynomialBasis):
	r""" A tensor product polynomial basis constructed from Chebyshev polynomials

	`Chebyshev polynomials`_ are an orthogonal basis on the interval :math:`[-1,1]`
	with the weight :math:`(1 - x^2)^{-1/2}`:

	.. math::
		\phi_0(x) &= 1 \\ 
		\phi_1(x) &= x \\
		\phi_2(x) &= 2x^2 - 1 \\
		\phi_3(x) &= 4x^3 - 3x \\ 
		\vdots & \phantom{=}\vdots \\
		\phi_k(x) &= 2x \phi_{k-1}(x) - \phi_{k-2}(x)

	.. _Chebyshev polynomials: https://en.wikipedia.org/wiki/Chebyshev_polynomials
	"""
	def _vander(self, *args, **kwargs):
		return chebvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return chebder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(chebroots(*args, **kwargs))

class HermitePolynomialBasis(TensorProductPolynomialBasis):
	r""" A tensor product polynomial basis constructed from Hermite polynomials

	`Hermite polynomials`_ are an orthogonal basis on the interval :math:`(-\infty, \infty)`
	with weight :math:`e^{-x^2/2}`.  
	Following :func:`~numpy.polynomial.hermite`, we use physicists Hermite polynomials

	.. math::
		\phi_0(x) &= 1 \\ 
		\phi_1(x) &= 2x \\
		\phi_2(x) &= 4x^2 - 1 \\
		\phi_3(x) &= 8x^3 - 12x \\ 
		\vdots & \phantom{=}\vdots \\
		\phi_k(x) &= (-1)^k e^{x^2} \frac{\partial^k}{\partial x^k} e^{-x^2}.

	Here we scale the coordinates such that they have zero mean and unit variance
	to improve the conditioning of the generalized Vandermonde matrix.

	.. _Hermite polynomials: https://en.wikipedia.org/wiki/Hermite_polynomials
	"""
	def _vander(self, *args, **kwargs):
		return hermvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return hermder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(hermroots(*args, **kwargs))
	
	def _set_scale(self):
		self._mean = np.mean(self.X, axis = 0)
		self._std = np.std(self.X, axis = 0)
	
	def _scale(self, X):		
		return (X - self._mean[None,:])/self._std[None,:]/np.sqrt(2)

	def _inv_scale(self, X):
		return np.sqrt(2)*self._std[None,:]*X + self._mean[None,:]

class LaguerrePolynomialBasis(TensorProductPolynomialBasis):
	r""" A tensor product polynomial basis constructed from Laguerre polynomials

	`Laguerre polynomials`_ are an orthogonal basis on the interval :math:`[0, \infty)`
	with weight :math:`e^{-x}`.  

	.. math::
		\phi_0(x) &= 1 \\ 
		\phi_1(x) &= -x+1 \\
		\phi_2(x) &= \frac12(x^2 - 4x + 1) \\
		\phi_3(x) &= \frac16(-x^3 + 9x^2 - 18x + 6) \\ 
		\vdots & \phantom{=}\vdots \\
		\phi_k(x) &= \sum_{\ell=0}^k {k \choose \ell} \frac{ (-1)^\ell}{\ell!} x^\ell.

	Here we scale the coordinates to best approximate a unit 
	exponential distribution. 

	.. _Laguerre polynomials: https://en.wikipedia.org/wiki/Laguerre_polynomials
	"""
	def _vander(self, *args, **kwargs):
		return lagvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return lagder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(lagroots(*args, **kwargs))

	def _set_scale(self):
		r""" Laguerre polynomial expects x[i] to be distributed like exp[-lam*x] on [0,infty)
		so we shift so that all entries are positive and then set a scaling
		"""
		self._lb = np.min(self.X, axis = 0)
		self._a = 1./np.mean(self.X - self._lb[None,:], axis = 0)
		
	def _scale(self, X):
		return self._a[None,:]*(X - self._lb[None,:])
		
	def _inv_scale(self, X):
		return X/self._a[None,:] + self._lb[None,:]





	
