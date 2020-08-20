import numpy as np
from polyrat import *
from polyrat.rational_ratio import _rational_residual_real, _rational_jacobian_real
from polyrat.rational_ratio import _rational_residual_complex, _rational_jacobian_complex
from polyrat.rational_ratio import _rational_residual_squared_abs_complex, _rational_jacobian_squared_abs_complex
from polyrat.rational_ratio import _rational_ratio_inf_complex

from .checkjac import *
from .test_data import *
import pytest


def test_rational_jacobian_real_arnoldi():
	r""" Test with high order approximation in the context of refinement
	"""
	M = 1000
	X, y = absolute_value(M)

	num_degree = 20
	denom_degree = 20

	sk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 5)
	sk.fit(X, y)
	
	P = sk.P
	Q = sk.Q	

	res = lambda x: _rational_residual_real(x, P, Q, y)
	jac = lambda x: _rational_jacobian_real(x, P, Q)

	a0 = sk.a
	b0 = sk.b
	x0 = np.hstack([a0, b0])

	err = check_jacobian(x0, res, jac, relative  =True)
	print(f"maximum error {err:8.2e}")
	assert err < 1e-8, "Error in the Jacobian"

def test_rational_jacobian_complex_arnoldi():
	M = 1000
	X, y = absolute_value(M, complex_ = True)

	num_degree = 20
	denom_degree = 20

	sk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 5)
	sk.fit(X, y)
	
	P = sk.P
	Q = sk.Q	

	res = lambda x: _rational_residual_complex(x, P, Q, y)
	jac = lambda x: _rational_jacobian_complex(x, P, Q)

	a0 = sk.a
	b0 = sk.b
	x0 = np.hstack([a0, b0])

	err = check_jacobian(x0.view(float), res, jac, relative = True)
	print(f"maximum error {err:8.2e}")
	assert err < 1e-8, "Error in the Jacobian"



@pytest.mark.parametrize("M", [500])
@pytest.mark.parametrize("dim", [1,2,3])
@pytest.mark.parametrize("num_degree", [4, 7])
@pytest.mark.parametrize("denom_degree", [4, 8, 9])
@pytest.mark.parametrize("complex_", [True, False])
@pytest.mark.parametrize("seed", [0])
def test_rational_jacobian(M, dim, num_degree, denom_degree, complex_, seed):
	
	X, y = random_data(M, dim, complex_, seed)

	P = LegendrePolynomialBasis(X, num_degree).basis()
	Q = LegendrePolynomialBasis(X, denom_degree).basis()

	x0 = np.random.randn(P.shape[1]+Q.shape[1])

	if complex_:
		res = lambda x: _rational_residual_complex(x, P, Q, y)
		jac = lambda x: _rational_jacobian_complex(x, P, Q)
		x0 = x0 + 1j*np.random.randn(P.shape[1]+Q.shape[1])
	else:	
		res = lambda x: _rational_residual_real(x, P, Q, y)
		jac = lambda x: _rational_jacobian_real(x, P, Q)
	
	err = check_jacobian(x0.view(float), res, jac, relative = True)
	print(f"maximum error {err:8.2e}")
	assert err < 1e-5, "Error in the Jacobian"



def test_rational_residual_squared_abs_complex():
	M = 1000
	dim = 2
	num_degree = 3
	denom_degree = 4

	X, y = random_data(M, dim, complex_ = True, seed = 0)

	P = LegendrePolynomialBasis(X, num_degree).basis()
	Q = LegendrePolynomialBasis(X, denom_degree).basis()

	x = np.random.randn(2*(P.shape[1]+Q.shape[1]))

	res = lambda x: _rational_residual_squared_abs_complex(x, P, Q, y)
	jac = lambda x: _rational_jacobian_squared_abs_complex(x, P, Q, y)
	
	err = check_jacobian(x, res, jac, relative = True)
	assert err < 1e-7


def test_rational_ratio_inf_complex():
	M = 1000
	dim = 1
	complex_ = True
	seed = 0
	num_degree = 5
	denom_degree = 5
	
	M = 1000
	#X, y = random_data(M, dim, complex_, seed)
	X, y = absolute_value(M, complex_)

	sk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 5, rebase= True)
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
	#test_rational_jacobian(1000, 2, 4, 5, True, 0)	
	test_rational_inf_constraint_complex()
