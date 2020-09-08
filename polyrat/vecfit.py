import numpy as np
from .aaa import _build_cauchy
from .arnoldi import ArnoldiPolynomialBasis
from iterprinter import IterationPrinter

def vecfit(X, y, num_degree, denom_degree, maxiter = 50, verbose = True, Basis = ArnoldiPolynomialBasis, poles0 = linearized):
	r"""Implements Vector Fitting 
	
	See: GS99

	Parameters
	----------

	poles0: ['GS', 'linearized', array-like]
		Specifies how the initial poles are to be selected
		* 'GS': as recommeneded by Gustavsen & Semlyen, linearized spaced between largest imaginary value
		* 'linearized', perform a linearized rational fitting
		* array-like: specify an array of denom_degree initial poles
	"""
	assert num_degree >= 0 and denom_degree >= 0, "numerator and denominator degrees must be nonnegative integers"
	assert num_degree +1 >= denom_degree, "Vector fitting requires denominator degree to be at most one less than numerator degree"

	if lam0 == 'GS':
		# Generate initial poles as recommened in GS99, Sec. 3.2 (eqns. 9-10)
		im_max = np.max(np.abs(x.imag))
		poles = -im_max/100 + 1j*np.linspace(-im_max, im_max, denom_degree) 
	else:
		assert len(poles0) == denom_degree, "Number of poles must match the degree of the denominator"
		poles = np.array(poles)		

	for it in range(maxiter):
		pass
	
	 
