import numpy as np


def abs_fun(X):
	X = np.array(X)
	assert X.shape[1] == 1, "Must be 1-dimensional input"
	return np.abs(X).flatten()


def penzl(X):
	r""" A one-parameter SISO transfer function example from Penzl 2006 (Example 3)

	Recommended parameter range: 1j*[1e1, 1e4]
	
	Parameters
	----------
	X: array-like (M, 1)
		Input values for the transfer function
	
	Returns
	-------
	H: np.array (M,)
		values the transfer function takes at the given points
	"""

	b = np.ones(1006)
	b[0:6] *= 10
	c = np.copy(b)

	A1 = np.array([[-1,100],[-100,-1]])
	A2 = np.array([[-1, 200],[-200,-1]])
	A3 = np.array([[-1, 400], [-400,-1]])
	A4 = -np.diag(np.arange(1, 1001))

	H = np.zeros(len(X), dtype = np.complex)
	for i, x in enumerate(X):
		z = x[0]

		# This is what we would be naively solving
		# A = block_diag(A1, A2, A3, A4)
		# H[i] = c.T @ np.linalg.solve(z*np.eye(1006) - A, b)

		# However, due to the block-diagonal structure, we can solve this
		# much more efficiently
		# Solve the 2x2 systems in the first three blocks
		H[i] += c[0:2].T @ np.linalg.solve(z*np.eye(2) - A1, b[0:2])
		H[i] += c[2:4].T @ np.linalg.solve(z*np.eye(2) - A2, b[2:4])
		H[i] += c[4:6].T @ np.linalg.solve(z*np.eye(2) - A3, b[4:6])
		# As this is purely diagonal, we directly invert
		H[i] += c[6:].T @ (1./(z - np.diag(A4))).reshape(-1,1) 

	return H
