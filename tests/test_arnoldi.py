import numpy as np
from polyrat import *
import pytest


@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_arnoldi_roots(n):
	r""" Check root computation in Arnoldi polynomials
	"""
	true_roots = np.arange(1, n+1)

	def wilkinson(x):
		value = np.zeros(x.shape, dtype = complex)
		for i, xi in enumerate(x):
			value[i] = np.prod(xi - true_roots)
		return value

	# It is important that we sample at the roots to avoid the large
	# values the Wilkinson polynomial takes away from these points.
	X = np.arange(0, n+1, step = 0.1, dtype = float).reshape(-1,1)
	y = wilkinson(X).flatten()
	
	arn = PolynomialApproximation(n, Basis = ArnoldiPolynomialBasis)
	arn.fit(X, y)
	roots = arn.roots().flatten()
	I = hungarian_sort(true_roots, roots)
	roots = roots[I]	
	print("true_roots", true_roots)
	print("roots", roots)
	print("value", arn(roots.reshape(-1,1)))
	for tr, r, fr in zip(true_roots, roots, arn(roots.reshape(-1,1))):	
		print(f'true root: {tr.real:+10.5e} {tr.imag:+10.5e}I \t root: {r.real:+10.5e} {r.imag:+10.5e} I \t abs fun value {np.abs(fr):10.5e}')

	print("computed roots", roots)
	print("true roots    ", true_roots)
	err = sorted_norm(roots, true_roots, np.inf)
	print("error", err)	
	assert err < 1e-7, "Error too large"


def test_arnoldi_vandermonde():
	dim = 2
	degree = 3
	X = np.random.randn(100, dim)
	arn = ArnoldiPolynomialBasis(X, degree)
	
	err = arn.vandermonde_X - arn.vandermonde(X)
	norm_err = np.linalg.norm(err, 'fro')
	assert norm_err < 1e-10
	
	# Note, these are already verified in test_basis -> test_vandermonde_derivative
	# so the following checks are simply that the pass through access works correctly 
	arn.vandermonde_derivative(X)
	arn.vandermonde_derivative(np.random.randn(10, dim))



if __name__ == '__main__':
	test_arnoldi_roots(20)
