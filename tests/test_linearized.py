import numpy as np
import pytest
from polyrat import *
try:
	from test_data import array_absolute_value
except ImportError:
	from .test_data import array_absolute_value


@pytest.mark.parametrize("degree", [(5,5), (10,10), (20,20)])
def test_linearized_ratfit(degree):
	M = 1000
	X = np.exp(1j*np.linspace(0, 2*np.pi, M, endpoint = False)).reshape(-1,1)
	y = np.tan(1*X.flatten())

	p, q = linearized_ratfit(X, y, degree[0], degree[1])
	
	err1 = np.linalg.norm(p(X)/q(X) - y)
	print(err1)

	# Monomials are well conditioned on the unit circle
	p, q = linearized_ratfit(X, y, degree[0], degree[1], Basis = MonomialPolynomialBasis)
	
	err2 = np.linalg.norm(p(X)/q(X) - y)
	print(err2)
	
	assert np.isclose(err1, err2)


	

if __name__ == '__main__':
	#test_linearized_ratfit((5,5))
	test_array( (2,1) )
	
