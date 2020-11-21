import numpy as np


def absolute_value(M, complex_ = True):
	r""" Generate test data for absolute value test function
	"""

	if complex_:
		X, Y = np.meshgrid(*[np.linspace(-1,1,int(np.sqrt(M))) for i in range(2)])
		X = X.reshape(-1,1) + 1j* Y.reshape(-1,1)
	else:
		X = np.linspace(-1,1, M).reshape(-1,1)
	
	y = np.abs(X).flatten()

	return X, y

def array_absolute_value(M, output_dim = ()):
	X = np.linspace(-1,1, M)
	
	if len(output_dim) == 0:
		x0 = 0
		return X.reshape(-1,1), np.abs(X - x0)
	else:
		x0 = np.linspace(-0.5,.5,int(np.prod(output_dim)))

	Y = np.zeros((M, *output_dim))

	for j, idx in enumerate(np.ndindex(output_dim)):
		Y[(slice(M), *idx)] = np.abs(X - x0[j])


	return X.reshape(-1,1), Y


def random_data(M, dim, complex_, seed ):
	np.random.seed(seed)
	X = np.random.randn(M, dim)
	y = np.random.randn(M)
	if complex_:
		X = X + 1j*np.random.randn(M, dim)
		y = y + 1j*np.random.randn(M)

	return X, y

