import numpy as np
from polyrat import *


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


if __name__ == '__main__':
	test_vecfit()

