import numpy as np
from polyrat import *
from polyrat.rat import _rational_residual_real, _rational_jacobian_real
#from polyrat.rat import _minimize_1_norm

#import cvxpy as cp

#def test_minimize_1_norm():
#	np.random.seed(0)
#	A = np.random.randn(100,50)
#	
#	x, s = _minimize_1_norm(A)
#
#
#	x_ = cp.Variable(A.shape[1])
#	prob = cp.Problem(cp.Minimize(cp.norm(A @ x_, 1)), [x_[0] == 1.])
#	prob.solve(verbose = True, solver = 'CVXOPT')
#
#	xt = x_.value
#	xt /= np.linalg.norm(xt)
#	xt *= np.sign(xt[0])
#	print(x)
#	print(xt)
#	# Check the nominal solution


		

	


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
	#test_skfit_rebase()
	#test_minimize_1_norm()
	#test_skfit()
	test_rational_jacobian_real()
