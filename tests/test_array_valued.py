import numpy as np
import pytest
from polyrat import *
try:
	from test_data import array_absolute_value
except ImportError:
	from .test_data import array_absolute_value



@pytest.mark.parametrize("output_dim", [(), (1,), (2,), (2,1), (2,2,2)])
@pytest.mark.parametrize("RationalApproximation",
	[LinearizedRationalApproximation,
	SKRationalApproximation,
	VectorFittingRationalApproximation,	
	StabilizedSKRationalApproximation,
	])
def test_array_valued(output_dim, RationalApproximation):
	r""" This mainly checks the functionality with array valued data
	"""
	num_degree = 10
	denom_degree = 10
	M = 1000
	X, Y = array_absolute_value(M, output_dim)

	rat =RationalApproximation(num_degree, denom_degree)
	rat.fit(X, Y)
	print(np.linalg.norm( (rat(X) - Y).flatten(), 2)/np.linalg.norm(Y.flatten(), 2))

if __name__ == '__main__':
	test_array_valued((5,), StabilizedSKRationalApproximation)
