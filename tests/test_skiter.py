

import numpy as np
from polyrat import *
from polyrat.skiter import _minimize_inf_norm_real, _minimize_inf_norm_complex

import pytest
from .test_data import *

def test_minimize_inf_norm_real():
	np.random.seed(0)
	A = np.random.randn(100, 10)
	x, cond = _minimize_inf_norm_real(A)
	
	assert np.isclose(np.linalg.norm(x,2),1)

	# Compare best solution against random solutions
	for i in range(1000):
		xi = np.random.randn(*x.shape)
		xi /= np.linalg.norm(xi)
		assert np.linalg.norm(A @ x, np.inf) <= np.linalg.norm(A @ xi, np.inf)

def test_minimize_inf_norm_complex():
	np.random.seed(0)
	A = np.random.randn(100, 10) + 1j*np.random.randn(100,10)
	x, cond = _minimize_inf_norm_complex(A, nsamp = 360)
	
	assert np.isclose(np.linalg.norm(x,2),1)

	# Compare best solution against random solutions
	obj = np.linalg.norm(A @ x, np.inf)

	for i in range(1000):
		xi = np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape)
		xi /= np.linalg.norm(xi)

		obj_new = np.linalg.norm(A @ xi, np.inf)
		print(obj_new - obj)
		assert obj <= obj_new





@pytest.mark.parametrize("M", [1000])
@pytest.mark.parametrize("num_degree", [5, [3,4]])
@pytest.mark.parametrize("denom_degree", [5, [5,3]])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("Basis", 
	[LegendrePolynomialBasis,
	 ArnoldiPolynomialBasis])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("norm", [2])
@pytest.mark.parametrize("refine", [True, False])
@pytest.mark.parametrize("complex_", [True, False])

def test_skfit_exact(M, dim, num_degree, denom_degree, refine, norm, Basis, seed, complex_):
	r"""
	When the data is a *exactly* a rational function of the specified degree,
	SK iteration should recover it exactly (modulo conditioning issues)
	"""

	X, y = random_data(M, dim, complex_, seed)
	
	# Exit without error if testing a total degree problem
	try:
		num_degree = int(num_degree)
	except (TypeError, ValueError):
		if len(num_degree) != dim: return
	try:
		denom_degree = int(denom_degree)
	except (TypeError, ValueError):
		if len(denom_degree) != dim: return

	# Generate exact fit
	P = LegendrePolynomialBasis(X, num_degree).vandermonde_X
	Q = LegendrePolynomialBasis(X, denom_degree).vandermonde_X

	# coefficients
	a = np.random.randn(P.shape[1])
	b = np.random.randn(Q.shape[1])
	if complex_:
		a = a + 1j*np.random.randn(*a.shape)
		b = b + 1j*np.random.randn(*b.shape)
		
	y = (P @ a)/(Q @ b)

	sk = SKRationalApproximation(num_degree, denom_degree, 
		norm = norm, Basis = Basis, verbose = True)
	
	sk.fit(X, y)
	if refine:
		sk.refine(X, y)
	
	err = np.linalg.norm(sk(X) - y)
	print(f" error : {err:8.2e}")
	assert err < 5e-8, "Expected an exact fit"



if __name__ == '__main__':
	#test_minimize_inf_norm_complex()	
	test_skfit_rebase()
