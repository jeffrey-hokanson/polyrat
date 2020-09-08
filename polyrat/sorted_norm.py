import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def hungarian_sort(a, b):
	r""" Plug in replacement for marriage_sort that uses Hungarian algorithm for an optimal pairing.

	"""
	a = np.array(a).flatten().reshape(-1,1)
	b = np.array(b).flatten().reshape(-1,1)
	assert a.shape == b.shape, "a and b must be the same shape"
	X = cdist(np.hstack([a.real, a.imag]), np.hstack([b.real, b.imag]))
	row, col = linear_sum_assignment(X)
	I = np.argsort(row)
	return col[I]	

def sorted_norm(a, b, ord=2):
	r""" Compute the norm of the optimaly sorted vectors

	"""
	I = hungarian_sort(a, b)	
	return np.linalg.norm(a - b[I], ord)
