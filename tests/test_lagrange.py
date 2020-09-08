import numpy as np
from polyrat import *


@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_roots_wilkinson(n):
	r""" Test computation of barycentric polynomial roots using the Wilkinson polynomial

	This test is the first one 
	"""
	true_roots = np.arange(1, n+1)
	def wilkinson(x):
		value = np.zeros(x.shape, dtype = np.complex)
		for i, xi in enumerate(x):
			value[i] = np.prod(xi - true_roots)
		return value


	# First try with basis sampling at nodes
	X = np.arange(0,n+1)
	lpi = LagrangePolynomialInterpolant(X, wilkinson(X))
	roots = lpi.roots()
	print(roots)		
	print(true_roots)
	err = sorted_norm(roots, true_roots, np.inf)	
	print(err)
	assert err < 1e-10, "Could not determine roots accurately"

	# Now spaced around a scaled unit circle (see  LC14, Sec.~4.1 )
	X = n*np.exp(1j*np.linspace(0,2*np.pi, n+1, endpoint = False))
	lpi = LagrangePolynomialInterpolant(X, wilkinson(X))
	roots = lpi.roots()
	print(roots)		
	print(true_roots)
	err = sorted_norm(roots, true_roots, np.inf)	
	print(err)
	assert err < 1e-10, "Could not determine roots accurately"

if __name__ == '__main__':
	test_roots_wilkinson(5)
