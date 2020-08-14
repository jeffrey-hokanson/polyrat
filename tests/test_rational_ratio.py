import numpy as np
from polyrat import *
from polyrat.rational_ratio import _rational_residual_real, _rational_jacobian_real
from polyrat.rational_ratio import _rational_residual_complex, _rational_jacobian_complex

from checkjac import *
from test_data import *



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

	err = check_jacobian(x0, res, jac)
	assert err < 1e-6, "Error in the Jacobian"


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

	err = check_jacobian(x0.view(float), res, jac)
	assert err < 1e-5, "Error in the Jacobian"



if __name__ == '__main__':
	test_rational_jacobian_complex_arnoldi()	
