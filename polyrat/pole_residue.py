r""" Pole-residue rational function parameterization
"""
from itertools import product

import numpy as np
from .aaa import _build_cauchy as build_cauchy

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
