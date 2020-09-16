import numpy as np
import pytest
from polyrat import *



@pytest.mark.parametrize("Basis", 
	[MonomialPolynomialBasis,
	 LegendrePolynomialBasis,
	 ChebyshevPolynomialBasis,
	 HermitePolynomialBasis, 
	 LaguerrePolynomialBasis,
	 ArnoldiPolynomialBasis])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])
def test_wilkinson_roots(Basis, n):
	r""" Check root computation in Arnoldi polynomials
	"""

	if Basis in [LaguerrePolynomialBasis, HermitePolynomialBasis] and n>= 8:
		# These tests fail due to the ill-conditioning of this basis
		return

	true_roots = np.arange(1, n+1)

	def wilkinson(x):
		value = np.zeros(x.shape, dtype = np.complex)
		for i, xi in enumerate(x):
			value[i] = np.prod(xi - true_roots)
		return value

	# It is important that we sample at the roots to avoid the large
	# values the Wilkinson polynomial takes away from these points.
	X = np.arange(0, n+1, step = 0.1, dtype = np.float).reshape(-1,1)
	y = wilkinson(X)
	poly = PolynomialApproximation(n, Basis = Basis)
	poly.fit(X, y)
	roots = poly.roots().flatten()
	I = hungarian_sort(true_roots, roots)
	roots = roots[I]	
	for tr, r, fr in zip(true_roots, roots, poly(roots.reshape(-1,1))):	
		print(f'true root: {tr.real:+10.5e} {tr.imag:+10.5e}I \t root: {r.real:+10.5e} {r.imag:+10.5e} I \t abs fun value {np.abs(fr):10.5e}')

	err = sorted_norm(roots, true_roots, np.inf)
	print("error", err)	
	assert err < 1e-7, "Error too large"

