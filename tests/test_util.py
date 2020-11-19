import numpy as np
from polyrat import *
from polyrat.util import *

import pytest

@pytest.mark.parametrize("output_dim",[ (), (1,), (2,), (2,1), (2,2,1)])
@pytest.mark.parametrize("complex_", [True, False])
def test_linearized(output_dim, complex_):
	#complex_ = True
	M = 50
	#output_dim = ()
	if complex_:
		X = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
	else:
		X = np.random.randn(M, 1)

	P = LegendrePolynomialBasis(X, 2).vandermonde_X
	Q = LegendrePolynomialBasis(X, 2).vandermonde_X

	Y = np.random.randn(M, *output_dim)
	A = linearized_ratfit_operator_dense(P, Q, Y)

	As = LinearizedRatfitOperator(P, Q, Y)

	assert np.all(A.shape == As.shape), "Dimensions do not match"

	X = np.random.randn(A.shape[1], 3)
	diff = np.linalg.norm(A @ X - As @ X, 'fro')
	print("forward error", diff)
	assert np.all(np.isclose(A @ X, As @ X))

	# Now check the adjoints
	Z = np.random.randn(A.shape[0], 3)

	diff = np.linalg.norm( As.adjoint() @ Z - A.conj().T @ Z, 'fro')
	print("adjoint error", diff)
	assert np.all(np.isclose(As.adjoint() @ Z, A.conj().T @ Z))

if __name__ == '__main__':
	pass
#	test_linearized()	
