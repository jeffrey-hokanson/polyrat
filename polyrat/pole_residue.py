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
	Cp = C**2

	# derivative with respect to lambda
	#J = [np.kron(a, Cp[:,j]).flatten() for j in range(len(lam))] 
	#J = np.column_stack(J)
	#J = [np.kron(a, Cp[:,j]).flatten() for j in range(len(lam))] 
	#J = np.einsum('ij,j...->i...j', C, a).reshape(-1,len(lam))

	# Derivative with respect to lambda
	J = [-np.kron(Cp[:,j], a[j]).flatten() for j in range(len(lam))]

	output_dim = a.shape[1:]

	if len(output_dim) == 0:
		J += [ -C, -V]
	else:
		dim = int(np.prod(output_dim))
		J += [ -np.kron(C[:,j], e.reshape(*a.shape[1:])).flatten() 
				for j, e in product(range(len(lam)), np.eye(dim))
			]
		J += [ -np.kron(V[:,j], e.reshape(*d.shape[1:])).flatten() 
				for j, e in product(range(V.shape[1]), np.eye(dim))
			]
	jacobian = np.column_stack(J)

	return residual, jacobian
