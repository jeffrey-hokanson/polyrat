import numpy as np
from polyrat import *
import pytest 


@pytest.mark.parametrize("n", [5, 10, 15, 20])
@pytest.mark.parametrize("ang", [1, (1 +1j)/np.sqrt(2)])
@pytest.mark.parametrize("deflate", [True, False])
def test_wilkinson(n, ang, deflate):
	r""" Test computation of barycentric polynomial roots using the Wilkinson polynomial

	This test is the first one 
	"""
	if deflate:
		tol = 1e-12
	else:
		tol = 1e-10

	true_roots = ang* np.arange(1, n+1)
	def wilkinson(x):
		value = np.zeros(x.shape, dtype = complex)
		for i, xi in enumerate(x):
			value[i] = np.prod(xi - true_roots)
		return value


	# First try with basis sampling at nodes
	X = ang*np.arange(0,n+1)
	lpi = LagrangePolynomialInterpolant(X, wilkinson(X))
	roots = lpi.roots(deflate)
	print(roots)		
	print(true_roots)
	err = sorted_norm(roots, true_roots, np.inf)	
	print(err)
	assert err < 1e-10, "Could not determine roots accurately"

	# Now shifted slightly off the poles (see  LC14, Sec.~4.4 )
	#X = (n+1)*np.exp(1j*np.linspace(0,2*np.pi, n+1, endpoint = False))
	X = ang*np.arange(0.5, 1.5 +n)
	lpi = LagrangePolynomialInterpolant(X, wilkinson(X))
	roots = lpi.roots(deflate)
	print(roots)		
	print(true_roots)
	print(true_roots - roots)
	err = sorted_norm(roots, true_roots, np.inf)	
	print(err)
	assert err < 1e-10, "Could not determine roots accurately"


	# Now test evaluation
	X = ang*np.linspace(0, n+1, 500)
	fX_true = wilkinson(X)
	fX = lpi(X)
	err = np.linalg.norm(fX_true - fX)
	rel_err = err/np.linalg.norm(fX_true)
	print("relative error in evaluation", rel_err)
	assert rel_err < 1e-10, "Error in evaluating Vandermonde matrix"

	# Test evaluation on nodes
	fX = lpi(lpi.nodes)
	err = np.linalg.norm(lpi.coef - fX)
	rel_err = err/np.linalg.norm(lpi.coef)
	print("relative error in evaluation", rel_err)
	assert rel_err < 1e-10, "Error in evaluating Vandermonde matrix"



if __name__ == '__main__':
	test_wilkinson(20, 1, True)
