import numpy as np
from polyrat import *
try:
	from .test_data import *
except ImportError:
	from test_data import *

import pytest


@pytest.mark.parametrize("Basis",
	[MonomialPolynomialBasis,
	 LegendrePolynomialBasis,
	 ChebyshevPolynomialBasis,
	 HermitePolynomialBasis, 
	 LaguerrePolynomialBasis,
	 ArnoldiPolynomialBasis])

def test_rational_ratio(Basis):
	seed = 0
	X, y1 = random_data(1000, 1, complex_ = False, seed = seed)
	y2 = np.random.randn(*y1.shape)	
	
	p = PolynomialApproximation(5, Basis)
	p.fit(X, y1)
	q = PolynomialApproximation(6, Basis)
	q.fit(X, y2)

	rat = RationalRatio(p, q)

	poles, residues = rat.pole_residue()
	print("poles", poles)
	print("residues", residues)


def test_rational_pole_residue():
	X, Y = array_absolute_value(1000, (5,4))
	rat = SKRationalApproximation(5,6, Basis = LegendrePolynomialBasis)
	rat.fit(X, Y)
	
	poles, residues = rat.pole_residue()
	xhat = X[0:1]
	yhat = rat(xhat)
	yhat2 = np.zeros(yhat.shape, dtype = complex)
	for lam, R in zip(poles, residues):
		yhat2 += R/(xhat.flatten() - lam)

	print("true")	
	print(yhat)
	print("sum")
	print(yhat2)
	print(np.isclose(yhat, yhat2))
	assert np.all(np.isclose(yhat, yhat2))

if __name__ == '__main__':
	test_rational_pole_residue()
	#test_rational_ratio(MonomialPolynomialBasis)
	#test_skfit_rebase()
	#test_minimize_1_norm()
	#test_skfit()
