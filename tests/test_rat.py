import numpy as np
from polyrat import *


def test_skfit_rebase():
	np.random.seed(0)
	M = 500
	dim = 2
	X = np.random.randn(M, dim)


	num_degree = [4,8]
	denom_degree = [4, 8]

	P = LegendrePolynomialBasis(X, num_degree).basis() 
	Q = LegendrePolynomialBasis(X, denom_degree).basis() 
	
	a = np.random.randn(P.shape[1])
	b = np.random.randn(Q.shape[1])

	g = 1e-3*np.random.randn(M) # Noise
	y = (P @ a)/(Q @ b) + g
	print(np.linalg.norm(g))
	
	skfit_rebase(X, y, num_degree, denom_degree, norm = 1)


if __name__ == '__main__':
	test_skfit_rebase()
