import numpy as np
from polyrat import *

def test_skfit():
	sk = SKRationalApproximation(20,20, refine = True, rebase = True, norm = np.inf)

	X = np.linspace(-1,1, int(2e3)).reshape(-1,1)
#	X = 1j*X
	y = np.abs(X).flatten()
	sk.fit(X, y)

	print("sup norm error", np.linalg.norm(sk(X) - y, np.inf))


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
	pass
	#test_skfit_rebase()
	#test_minimize_1_norm()
	#test_skfit()
