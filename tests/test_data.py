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

def random_data(M, dim, complex_, seed ):
	np.random.seed(seed)
	X = np.random.randn(M, dim)
	y = np.random.randn(M)
	if complex_:
		X = X + 1j*np.random.randn(M, dim)
		y = y + 1j*np.random.randn(M)

	return X, y

