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

	print("poles", rat.poles())

	

if __name__ == '__main__':
	test_rational_ratio(MonomialPolynomialBasis)
	#test_skfit_rebase()
	#test_minimize_1_norm()
	#test_skfit()
