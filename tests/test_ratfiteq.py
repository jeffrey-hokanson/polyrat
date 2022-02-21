import numpy as np
from polyrat.ratfiteq import *
from checkjac import *

def test_ratfiteq():
	M = 1000
	x = np.linspace(-1,1, M).reshape(-1,1)
	y = np.abs(x).flatten()

	num_degree = [6]
	denom_degree = [6]
	con = ConstantRationalConstraint(x, y, num_degree, denom_degree)

	a = np.random.randn(num_degree[0]+1)
	b = np.random.randn(denom_degree[0]+1)
	yfit = y + 1e-4*np.random.randn(*y.shape)
	r = con.fun_native(a, b, yfit)	
	J = con.jac_native(a, b, yfit)	

	z = con.coding.encode(a, b, yfit)

	err = check_jacobian(z, con.fun, con.jac)	
	assert err < 1e-7

	U = con.orthogonal_nullspace(con.jac(z))
	err = J @ U
	print('nullspace', np.max(np.abs(err)))
	assert np.max(np.abs(err)) < 1e-10

	obj = RationalObjective(y, con.coding)

	err = check_jacobian(z, obj.fun, obj.jac) 
	assert err < 1e-8
	err = check_jacobian(z, obj.jac, obj.hess) 
	assert err < 1e-8

	fit = RatFitEquality(x, y, num_degree, denom_degree)
	fit.solve()

if __name__ == '__main__':
	test_ratfiteq()
