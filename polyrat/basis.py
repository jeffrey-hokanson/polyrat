r""" Basic polynomial classes

These classes use the polynomial representations included in numpy

"""

import numpy as np
from numpy.polynomial.polynomial import polyvander, polyder, polyroots
from numpy.polynomial.legendre import legvander, legder, legroots 
from numpy.polynomial.chebyshev import chebvander, chebder, chebroots
from numpy.polynomial.hermite import hermvander, hermder, hermroots
from numpy.polynomial.laguerre import lagvander, lagder, lagroots
from functools import lru_cache
from .index import *

class PolynomialBasis:

	def basis(self):
		r""" Alias for vandermonde(X) where X is the points where the basis was initialized

		Returns
		-------
		V: np.array (M, N)
			generalized Vandermonde matrix evaluated using the points the basis was initialized with
		"""
		raise NotImplementedError
	
		
	def vandermonde(self, X):
		r""" Construct the generalized Vandermonde matrix associated with the polynomial basis 

		Parameters
		----------
		X: np.array (M, dim)
			Coordinates at which to evaluate the basis at 

		Returns
		-------
		V: np.array (M, N)
			generalized Vandermonde matrix 
		"""
		raise NotImplementedError

	def vandermonde_derivative(self, X):
		raise NotImplementedError


class TensorProductPolynomialBasis(PolynomialBasis):
	def _vander(self, *args, **kwargs):
		raise NotImplementedError
	def _der(self, *args, **kwargs):
		raise NotImplementedError
	def roots(self, *args, **kwargs):
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

	def _inv_scale(self, X):
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
		r""" Construct a column-wise derivative of the generalized Vandermonde matrix
		"""
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
		  
	

	@lru_cache(maxsize = 1)
	def basis(self):
		r""" The basis for the input coordinates
		""" 
		return self.vandermonde(self.X)
	
	@property
	@lru_cache(maxsize = 1)
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


	@property
	@lru_cache(maxsize = 1)
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
	def _vander(self, *args, **kwargs):
		return polyvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return polyder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(polyroots(*args, **kwargs))


class LegendrePolynomialBasis(TensorProductPolynomialBasis):
	def _vander(self, *args, **kwargs):
		return legvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return legder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(legroots(*args, **kwargs))

class ChebyshevPolynomialBasis(TensorProductPolynomialBasis):
	def _vander(self, *args, **kwargs):
		return chebvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return chebder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(chebroots(*args, **kwargs))

class HermitePolynomialBasis(TensorProductPolynomialBasis):
	def _vander(self, *args, **kwargs):
		return hermvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return hermder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(hermroots(*args, **kwargs))
	
	def _set_scale(self, X):
		self._mean = np.mean(X, axis = 0)
		self._std = np.std(X, axis = 0)
	
	def _scale(self, X):		
		return (X - self._mean[None,:])/self._std[None,:]/np.sqrt(2)

	def _inv_scale(self, X):
		return np.sqrt(2)*self._std[None,:]*X + self._mean[None,:]

class LaguerrePolynomialBasis(TensorProductPolynomialBasis):
	def _vander(self, *args, **kwargs):
		return lagvander(*args, **kwargs)
	def _der(self, *args, **kwargs):
		return lagder(*args, **kwargs)
	def roots(self, *args, **kwargs):
		return self._inv_scale(lagroots(*args, **kwargs))

	def _set_scale(self, X):
		r""" Laguerre polynomial expects x[i] to be distributed like exp[-lam*x] on [0,infty)
		so we shift so that all entries are positive and then set a scaling
		"""
		self._lb = np.min(X, axis = 0)
		self._a = 1./np.mean(X - self._lb[None,:], axis = 0)
		
	def _scale(self, X):
		return self._a[None,:]*(X - self._lb[None,:])
		
	def _inv_scale(self, X):
		return X/self._a[None,:] + self._lb[None,:]





	
