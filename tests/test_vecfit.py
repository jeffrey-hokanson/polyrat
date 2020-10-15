import numpy as np
from polyrat import *
from polyrat.demos import penzl


import pytest

def test_vecfit():
	X = np.linspace(-1, 1, int(2e3)).reshape(-1,1)
	y = np.abs(X).flatten()

	num_degree = 18
	denom_degree = 18
	poles0 = np.random.randn(denom_degree)
	a, b, poles, bonus_poly = vecfit(X, y, num_degree, denom_degree, poles0 = poles0, maxiter = 500)


	vf = VectorFittingRationalFunction(a, b, poles, bonus_poly)
	r = vf(X)	
	print("residual outside", np.linalg.norm(r - y))


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("degree", [(12,12), (14, 12),])
def test_init(verbose, degree):
	X = 1j*np.logspace(1, 3, int(2e3)).reshape(-1,1)
	X = np.vstack([X, X.conj()])
	y = penzl(X)
	
	num_degree = degree[0]
	denom_degree = degree[1]

	
	sk = SKRationalApproximation(num_degree, denom_degree)
	sk.fit(X, y)
	err_sk = np.linalg.norm(sk(X) - y)

	for init in ['linearized', 'GS', 'random']:
		if init == 'random':
			poles0 = np.random.randn(denom_degree)
		else:
			poles0 = init
		vf = VectorFittingRationalApproximation(num_degree, denom_degree, poles0 = poles0, verbose = verbose)

		vf.fit(X, y)
		err_vf = np.linalg.norm(vf(X) - y)
		print(init, 'vf', err_vf, 'sk', err_sk)
		assert np.isclose(err_vf, err_sk, atol = 1e-1, rtol = 1e-1) 	

if __name__ == '__main__':
	#test_vecfit()
	test_init(True, (12,12))

