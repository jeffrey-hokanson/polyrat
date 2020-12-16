r""" Pole-residue rational function parameterization
"""
from itertools import product

import numpy as np
from .aaa import _build_cauchy as build_cauchy
import scipy.optimize

def residual_jacobian_real(x, Y, V, lam, a, d, jacobian = False):
	r""" This implementation requires all data to be real: x, Y, lam, a, V, d.

	"""

	C = build_cauchy(x, lam)

	residual = Y.flatten() - np.einsum('ij,j...->i...', C, a).flatten()
	residual -= np.einsum('ij,j...->i...', V, d).flatten()

	if not jacobian: 
		return residual

	# Column-wise derivative of the Cauchy matrix	

	# Derivative with respect to lambda
	J = [-np.einsum('i,...->i...', C[:,j]**2, a[j]).flatten() for j in range(len(lam))]
	
	output_dim = a.shape[1:]

	if len(output_dim) == 0:
		J += [ -C, -V]
	else:
		dim = int(np.prod(output_dim))
		J += [ -np.einsum('i,...->i...', C[:,j], e.reshape(output_dim)).flatten() 
				for j, e in product(range(len(lam)), np.eye(dim))
			]
		J += [ -np.einsum('i,...->i...', V[:,j], e.reshape(output_dim)).flatten() 
				for j, e in product(range(V.shape[1]), np.eye(dim))
			]
	jacobian = np.column_stack(J)

	return residual, jacobian



def pole_residue_real(x, Y, V, lam0, a0, d0, stable = False, **kwargs):
	r"""

	
	Parameters
	----------
	x: array_like
		Input data of shape (M, 1)
	Y: array_like
		Output data of shape (M,...)
	V: array_like 
		Polynomial basis of shape (M, N).
		Note if there is no polynomial term, this should be an array of size (M,0)
	lam0: array_like
		Initial poles of shape (r,)
	a0: array_like
		Residues for each of the pole of shape (r, ...)
	d0: array_like
		Polynomial coefficients for polynomial terms of shape (N, ...)
	stable: bool
		If True, force poles to lie in left half plane; otherwise 
	**kwargs: dict
		Additional arguments to be passed to :class:`~scipy:scipy.optimize.least_squares` 

	Returns
	-------
	lam: :class:`~numpy:numpy.ndarray`
		Poles of shape (r,)
	a: :class:`~numpy:numpy.ndarray`
		Residues of shape (r, ...)
	d: :class:`~numpy:numpy.ndarray`
		Polynomial coefficients of shape (N, ...)
	"""	
	
	forward = lambda lam, a, d : np.hstack([lam, a.flatten(), d.flatten()])
	inverse = lambda xx: (
		xx[:len(lam0)], 
		xx[len(lam0):(len(lam0)+len(a0.flatten()))].reshape(a0.shape), 
		xx[(len(lam0)+len(a0.flatten())):].reshape(d0.shape)
		)


	res = lambda xx: residual_jacobian_real(x, Y, V, *inverse(xx), jacobian = False)
	jac = lambda xx: residual_jacobian_real(x, Y, V, *inverse(xx), jacobian = True)[1]


	xx0 = forward(lam0, a0, d0)

	if stable:
		bounds = [
			forward(-np.inf*np.ones(lam0.shape), -np.inf*np.ones(a0.shape), -np.inf*np.ones(d0.shape)),
			forward(0*np.ones(lam0.shape), np.inf*np.ones(a0.shape), np.inf*np.ones(d0.shape)),
			]
		
		result = scipy.optimize.least_squares(res, xx0, jac, bounds, **kwargs)
	else:	
		result = scipy.optimize.least_squares(res, xx0, jac, **kwargs)

	lam, a, d = inverse(result.x)

	return lam, a, d	
