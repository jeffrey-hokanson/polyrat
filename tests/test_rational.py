import numpy as np
from polyrat import *
from .test_data import *
import pytest


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
@pytest.mark.parametrize("rebase", [True, False])
@pytest.mark.parametrize("complex_", [True, False])

def test_skfit_exact(M, dim, num_degree, denom_degree, refine, norm, Basis, seed, complex_, rebase):
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
	P = LegendrePolynomialBasis(X, num_degree).basis()
	Q = LegendrePolynomialBasis(X, denom_degree).basis()

	# coefficients
	a = np.random.randn(P.shape[1])
	b = np.random.randn(Q.shape[1])
	if complex_:
		a = a + 1j*np.random.randn(*a.shape)
		b = b + 1j*np.random.randn(*b.shape)
		
	y = (P @ a)/(Q @ b)

	sk = SKRationalApproximation(num_degree, denom_degree, refine = refine, 
		norm = norm, Basis = Basis, rebase = rebase, verbose = True)
	
	sk.fit(X, y)
	
	err = np.linalg.norm(sk(X) - y)
	print(f" error : {err:8.2e}")
	assert err < 5e-8, "Expected an exact fit"

if __name__ == '__main__':
	test_skfit_exact(1000, 2, 5, 5, True, ArnoldiPolynomialBasis, 0, complex_ = False, rebase = True)
	pass
	#test_skfit_rebase()
	#test_minimize_1_norm()
	#test_skfit()
