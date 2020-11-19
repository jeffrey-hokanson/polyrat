import numpy as np
from polyrat import *
from polyrat.linratfit import _linearized_2norm


def test_linearized_2norm():

	X = np.random.randn(M, 1)
	P = LegendrePolynomialBasis(X, 3).vandermonde_X
	Q = LegendrePolynomialBasis(X, 4).vandermonde_X

	Y = np.random.randn(M, 2)

	a, b = _linearized_2norm(P, Q, Y)



if __name__ == '__main__':
	test_linearized_2norm()
