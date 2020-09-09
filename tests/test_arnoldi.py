import numpy as np
from polyrat import *
import pytest


@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_arnoldi_roots(n):
	r""" Check root computation in Arnoldi polynomials
	"""
	true_roots = np.arange(1, n+1)

	def wilkinson(x):
		value = np.zeros(x.shape, dtype = np.complex)
		for i, xi in enumerate(x):
			value[i] = np.prod(xi - true_roots)
		return value

	# It is important that we sample at the roots to avoid the large
	# values the Wilkinson polynomial takes away from these points.
	X = np.arange(0, n+1, step = 0.1, dtype = np.float).reshape(-1,1)
	y = wilkinson(X)
	arn = PolynomialApproximation(n, Basis = ArnoldiPolynomialBasis)
	arn.fit(X, y)
	roots = arn.roots()
	print("computed roots", roots)
	print("true roots    ", true_roots)
	err = sorted_norm(roots, true_roots, np.inf)
	print("error", err)	
	assert err < 1e-7, "Error too large"

if __name__ == '__main__':
	test_arnoldi_roots(15)
