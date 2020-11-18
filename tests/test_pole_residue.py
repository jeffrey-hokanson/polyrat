import numpy as np
from polyrat import *
from polyrat.pole_residue import *
from checkjac import *

import pytest


def slow_residual(x, Y, V, lam, a, c):
	r = np.copy(Y)

	M = len(x)

	if len(r.shape) == 1:
		idx_iter = [[],]
	else:
		idx_iter = np.ndindex(r.shape[1:])

	for idx in idx_iter:
		for i in range(M):
			for j in range(len(lam)):
				r[tuple([i, *idx])] -= a[tuple([j, *idx])]/(x[i] - lam[j])
			for j in range(len(c)):
				r[tuple([i, *idx])] -= c[tuple([j, *idx])] * V[i,j]

	return r.flatten()


@pytest.mark.parametrize("output_dim",[
		(),
		(1,),
		(2,),
		(3,1), 
		(3,2,1),
	])
def test_residual_jacobian_real(output_dim):
	np.random.seed(0)
	M = 10
	x = np.linspace(-1,1, M)
	Y = np.random.randn(M, *output_dim)

	lam = np.array([-0.1, -0.1])
	a = np.random.randn(len(lam), *output_dim)
	V = np.random.randn(M, 5)
	c = 0*np.random.randn(V.shape[1], *output_dim)

	r = residual_jacobian_real(x, Y, V, lam, a, c, jacobian = False) 
	r_true = slow_residual(x, Y, V, lam, a, c)

	err = r - r_true.flatten()
	print("fast", r[:5])
	print("slow", r_true[:5])
	err_norm = np.max(np.abs(err))
	print("residual error", err_norm)
	assert err_norm < 1e-10, "Residual did not compute the correct quantity"


	# Now check the residual

	forward = lambda lam, a, c : np.hstack([lam, a.flatten(), c.flatten()])
	inverse = lambda xx: (
		xx[:len(lam)], 
		xx[len(lam):(len(lam)+len(a.flatten()))].reshape(*a.shape), 
		xx[(len(lam)+len(a.flatten())):].reshape(c.shape)
		)

	xx0 = forward(lam, a, c)
	lam0, a0, c0 = inverse(xx0)
	assert np.all(np.isclose(lam0, lam))
	assert np.all(np.isclose(a0, a))
	assert np.all(np.isclose(c0, c))

	res = lambda xx: residual_jacobian_real(x, Y, V, *inverse(xx), jacobian = False)
	jac = lambda xx: residual_jacobian_real(x, Y, V, *inverse(xx), jacobian = True)[1]


	print("------- Checking the Jacobian ---------")

	err = check_jacobian(xx0, res, jac, relative = True)

	assert err < 1e-5, "inaccurate Jacobian"


if __name__ == '__main__':
	test_residual_jacobian_real((2,1))


