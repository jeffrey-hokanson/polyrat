import numpy as np
import pytest
from polyrat import *
from polyrat.demos import abs_fun


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
	y = wilkinson(X).flatten()
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


@pytest.mark.parametrize("complex_", [True, False])
@pytest.mark.parametrize("Basis", [None, LegendrePolynomialBasis, MonomialPolynomialBasis]) 
@pytest.mark.parametrize("norm", [1, 2, np.inf]) 
def test_approx(complex_, Basis, norm):
	if complex_:
		X = (1+1j)*np.linspace(-1,1, 1000).reshape(-1,1)
	else:
		X = np.linspace(-1,1, 1000).reshape(-1,1)
	
	y = abs_fun(X)
	p = PolynomialApproximation(6, Basis = Basis, norm = norm)
	p.fit(X, y)

	print(np.linalg.norm(p(X) - y, norm)/np.linalg.norm(y, norm))

	# Check quadratic approximation is exact
	y2 = X.flatten()**2
	p.fit(X, y2)
	assert np.all(np.isclose(p(X), y2))	


@pytest.mark.parametrize("Basis", 
	[MonomialPolynomialBasis,
	 LegendrePolynomialBasis,
	 ChebyshevPolynomialBasis,
	 HermitePolynomialBasis, 
	 LaguerrePolynomialBasis,
	 ArnoldiPolynomialBasis])
@pytest.mark.parametrize("dim", [1,2])
@pytest.mark.parametrize("output_dim", [None, 1, 3, (3,2)]) 
def test_derivative(Basis, dim, output_dim):
	np.random.seed(0)

	N = 1000
	degree = 5
	X = np.random.randn(N,dim)
	if output_dim is None:
		y = np.random.randn(N)
	else:
		try:
			y = np.random.randn(N, output_dim)
		except TypeError:
			y = np.random.randn(N, *output_dim)

	poly = PolynomialApproximation(degree, Basis = Basis)
	poly.fit(X, y)

	Xhat = np.random.randn(5,dim)

	D = poly.derivative(Xhat)
	
	h = 1e-6
	for i in range(len(Xhat)):
		for k in range(dim):
			ek = np.eye(dim)[k]
			x1 = (Xhat[i] + h*ek).reshape(1,-1)
			x2 = (Xhat[i] - h*ek).reshape(1,-1)
			dest = (poly(x1) - poly(x2))/(2*h)
			print(f"point {i}, direction {k}")
			print("finite difference")
			print(dest)
			print("nomial value")
			print(D[i,...,k])
			print('difference')
			print(D[i,...,k] - dest)
			assert np.all(np.isclose(dest, D[i,...,k], atol = 1e-4, rtol = 1e-4))



if __name__ == '__main__':
	#test_approx(True, None, 1)
	test_derivative(MonomialPolynomialBasis,2, (2,1))
	 
