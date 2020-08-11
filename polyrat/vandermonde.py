import numpy as np
from itertools import product


def fixed_degree_index(dim, degree):
	r""" Generate all combinations where sum == degree
	"""
	if dim == 1:
		return np.array([[degree]]).astype(np.int)
	else:
		idx = fixed_degree_index(dim - 1, degree)
		idx = np.hstack([np.zeros((idx.shape[0],1), dtype = np.int), idx])
		indices = [idx]
		for i in range(1, degree+1):
			idx = fixed_degree_index(dim - 1, degree - i) 
			idx = np.hstack([i*np.ones((idx.shape[0],1), dtype = np.int), idx])
			indices.append(idx)
		return np.vstack(indices)

def total_degree_index(dim, degree):
	r"""

	Parameters
	----------
	dim: int, positive
		Number of dimensions
	degree: int, nonnegative
		Total degree of the polynomial

	Returns
	-------
	indices: np.array (M, dim)
		indices of polynomials 
	"""

	assert dim > 0, "Dimension must be positive"
	assert degree >=0, "Degree must be nonnegative"
		
	if dim == 1:
		return np.arange(degree+1).reshape(-1,1)	

	indices = [fixed_degree_index(dim, k) for k in range(degree+1)]
	return np.vstack(indices)	


def max_degree_index(degree):
	r""" List the indices of a maximum degree polynomial



	Parameters
	----------
	degree: array-like
		

	Returns
	-------
	index: list of tuples
		List of indices ordered as needed for 
	"""

	return np.array([ids for ids in product(*[range(d+1) for d in degree])])




