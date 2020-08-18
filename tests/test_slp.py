import numpy as np
from polyrat import *
from polyrat.rational_ratio import _rational_residual_real, _rational_jacobian_real
from polyrat.rational_ratio import _rational_residual_complex, _rational_jacobian_complex
from polyrat.rational_ratio import _rational_residual_squared_abs_complex, _rational_jacobian_squared_abs_complex
from polyrat.rational_ratio import _rational_ratio_inf_complex

def test_rational_ratio_inf_complex():
	M = 10000
	dim = 1
	complex_ = True
	seed = 0
	num_degree = 10
	denom_degree = 10
	
	#M = 1000
	#X, y = random_data(M, dim, complex_, seed)
	#X, y = absolute_value(M, complex_)

	X, Y = np.meshgrid(*[np.linspace(-1,1,int(np.sqrt(M))) for i in range(2)])
	X = X.reshape(-1,1) + 1j* Y.reshape(-1,1)
	y = np.abs(X).flatten()
	
	sk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 20, rebase= True)
	sk.fit(X, y)
	print(sk.a)
	print(sk.b)

	# Compute the error for the SK iteration
	err_old = np.max(np.abs(sk(X) -y))

	a, b = _rational_ratio_inf_complex(y, sk.P, sk.Q, sk.a, sk.b)
	print(a)
	print(b)
	# This should have improved the fit
	err = np.max(np.abs( (sk.P @ a)/(sk.Q @ b) -y))

	print(f"old error {err_old:8.2e}")
	print(f"new error {err:8.2e}")
	assert err < err_old, "Optimization should have improved the solution"

if __name__ == '__main__':
	test_rational_ratio_inf_complex()
	
