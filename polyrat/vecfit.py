import numpy as np
from copy import deepcopy as copy
from .aaa import _build_cauchy
from .arnoldi import ArnoldiPolynomialBasis
from .skiter import linearized_ratfit
from .polynomial import LagrangePolynomialInterpolant, Polynomial
from .basis import MonomialPolynomialBasis
from .rational import RationalFunction, RationalRatio
from iterprinter import IterationPrinter
import scipy.linalg

def _solve_linearized_svd(num_basis, denom_basis, y):
	r"""
	"""

	# TODO: This is the SVD approach;
	# we should (1) implement this as a separate problem
	# and (2) make use of the algorithm from DGB15a to 
	# solve this problem for a vector/matrix valued	f
	A = np.hstack( [num_basis, -(denom_basis.T*y).T])
	U, s, VH = scipy.linalg.svd(A, full_matrices = False, compute_uv = True, lapack_driver = 'gesvd')
	x = VH.conj().T[:,-1]
	A_cond = s[0]/s[-1]

	# Split into numerator and denominator coefficients
	a = x[:num_basis.shape[1]]
	b = x[num_basis.shape[1]:]
	return a, b


def vecfit(X, y, num_degree, denom_degree, verbose = True, 
	Basis = ArnoldiPolynomialBasis, poles0 = 'linearized',
	maxiter = 50, ftol = 1e-7):
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
	assert num_degree + 1 >= denom_degree, "Vector fitting requires denominator degree to be at most one less than numerator degree"


	if verbose:
		printer = IterationPrinter(it = '4d', res = '20.10e', delta = '10.4e')
		printer.print_header(it = 'iter', res = 'residual norm', delta = 'Δ fit') 

	if isinstance(poles0, str):
		if poles0 == 'GS':
			# Generate initial poles as recommened in GS99, Sec. 3.2 (eqns. 9-10)
			im_max = np.max(np.abs(X.imag))
			poles = -im_max/100 + 1j*np.linspace(-im_max, im_max, denom_degree)
		elif poles0 == 'linearized':
			numerator, denominator = linearized_ratfit(X, y, num_degree, denom_degree)
			poles = denominator.roots() 
	else:
		assert len(poles0) == denom_degree, "Number of poles must match the degree of the denominator"
		poles = np.array(poles0)		


	# Construct the Vandermonde matrix for the remaining terms
	if num_degree - denom_degree >= 0:
		bonus_basis = Basis(X, num_degree - denom_degree)
		V = bonus_basis.basis()
	else:
		bonus_basis = None
		V = np.zeros((len(y),0))

	r_old = np.zeros(y.shape)

	for it in range(maxiter):
		C = _build_cauchy(X, poles)

		num_basis = np.hstack([C, V])
		denom_basis = np.hstack([np.ones((len(X), 1)), C])

		a, b = _solve_linearized_svd(num_basis, denom_basis, y)
		b_norm = np.linalg.norm(b)
		b /= b_norm
		a /= b_norm
	
		# Compute the rational approximation
		r = (num_basis @ a) / (denom_basis @ b)

		residual_norm = np.linalg.norm( (y - r).flatten(), 2)
		delta_norm = np.linalg.norm( (r_old - r).flatten(), 2)
		
	
		if verbose:
			printer.print_iter(it = it, res = residual_norm, delta = delta_norm)	

		if it == maxiter - 1:
			break

		if delta_norm < ftol:
			if verbose: print("terminated due to small change in approximation")
			break
		
		# Update roots only if we are going to continue the iteration
		# (if we update these we change the polynomial values)
		# This is the root finding approach that Gustavsen takes in Vector Fitting
		# See Gus06: eq. 5
		poles = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(len(poles)), b[1:]))

		r_old = r		


	if bonus_basis:
		print(a[len(poles):])
		bonus_poly = Polynomial(bonus_basis, a[len(poles):])
		print(V @ a[len(poles):] - bonus_poly(X))
	else:
		bonus_poly = Polynomial(MonomialPolynomialBasis(X, 0), 0*a[0:1])

	return a[:len(poles)], b, poles, bonus_poly	


class VectorFittingRationalFunction(RationalRatio):
	def __init__(self, a, b, poles, bonus_poly = None):
		self._a = np.copy(a)
		self._b = np.copy(b)
		self.poles = np.copy(poles)
	
		self.bonus_poly = copy(bonus_poly)

	def eval(self, X):
		C = _build_cauchy(X, self.poles)
		r = (C @ self._a)/(self._b[0] + C @ self._b[1:])
		if self.bonus_poly is not None:
			r += self.bonus_poly(X)

		return r

			

class VectorFittingRationalApproximation(VectorFittingRationalFunction):
	def __init__(self, num_degree, denom_degree, *args, **kwargs):
		self.num_degree = int(num_degree)
		self.denom_degree = int(denom_degree)
		self.args = args
		self.kwargs = kwargs
	
	def fit(self, X, y):
		self.a, self.b, self.poles, self.bonus_poly	= vecfit(X, y, *self.args, **self.kwargs)
