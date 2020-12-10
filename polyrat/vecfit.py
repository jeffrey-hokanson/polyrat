import numpy as np
from copy import deepcopy as copy
from .aaa import _build_cauchy
from .arnoldi import ArnoldiPolynomialBasis
from .linratfit import linearized_ratfit
from .polynomial import Polynomial
from .lagrange import LagrangePolynomialInterpolant
from .basis import MonomialPolynomialBasis, LegendrePolynomialBasis
from .rational import RationalFunction, RationalRatio
from iterprinter import IterationPrinter
import scipy.linalg

from .util import minimize_2norm_varpro, minimize_2norm_dense


def _solve_linearized_vecfit(num_basis, denom_basis, y):
	A = np.hstack([ num_basis, -(denom_basis[:,1:].T * y).T ])
	b = y
	x, res, rank, s = scipy.linalg.lstsq(A, b, overwrite_a = True, overwrite_b = False)
	a = x[0:num_basis.shape[1]]
	b = np.hstack([1, x[num_basis.shape[1]:]])
	return a, b, s[0]/s[-1]

# TODO: Implement a DGB15a variant of the linear solve that allows array-valued y

# TODO: Add matrix-weight option

def vecfit(X, y, num_degree, denom_degree, verbose = True, 
	Basis = ArnoldiPolynomialBasis, poles0 = 'linearized',
	maxiter = 50, ftol = 1e-10, btol = 1e-7):
	r"""Implements Vector Fitting 
	
	See: GS99

	Parameters
	----------
	X: numpy array  (M, m)
		Input coordinates
	y: numpy array (M,*)
		Output values

	poles0: ['GS', 'linearized', array-like]
		Specifies how the initial poles are to be selected
		* 'GS': as recommeneded by Gustavsen & Semlyen, linearized spaced between largest imaginary value
		* 'linearized', perform a linearized rational fitting
		* array-like: specify an array of denom_degree initial poles
	"""
	nout_dim = len(y.shape[1:])
	assert num_degree >= 0 and denom_degree >= 0, "numerator and denominator degrees must be nonnegative integers"
	assert num_degree + 1 >= denom_degree, "Vector fitting requires denominator degree to be at most one less than numerator degree"
	if isinstance(poles0, str):
		assert poles0 in ['linearized', 'GS']

	if verbose:
		printer = IterationPrinter(it = '4d', res = '20.10e', delta = '10.4e', bnorm = '10.4e', cond = '10.4e')
		printer.print_header(it = 'iter', res = 'residual norm', delta = 'Δ fit', bnorm = '‖b - e₀‖₂', cond = 'condion #') 

	if isinstance(poles0, str):
		if poles0 == 'linearized':
			# Generate initial poles by one step of SK iteration (i.e., the linearized ratfit)
			numerator, denominator = linearized_ratfit(X, y, num_degree, denom_degree, simultaneous = True)
			poles = denominator.roots().flatten()
		elif poles0 == 'GS':
			# Generate initial poles as recommened in GS99, Sec. 3.2 (eqns. 9-10)
			im_max = np.max(np.abs(X.imag))
			assert im_max > 0, "Must have a non-zero imaginary extent to use `poles0='GS'` initialization"
			poles = -im_max/100 + 1j*np.linspace(-im_max, im_max, denom_degree)
	else:
		assert len(poles0) == denom_degree, "Number of poles must match the degree of the denominator"
		poles = np.array(poles0)	

	# Construct the Vandermonde matrix for the remaining terms
	if num_degree - denom_degree >= 0:
		bonus_basis = Basis(X, num_degree - denom_degree)
		V = bonus_basis.vandermonde_X
	else:
		bonus_basis = None
		V = np.zeros((len(y),0))

	r_old = np.zeros(y.shape)

	best_fit = {'residual_norm':np.inf}

	for it in range(maxiter):
		C = _build_cauchy(X, poles)

		num_basis = np.hstack([C, V])
		denom_basis = np.hstack([np.ones((len(X), 1)), C])

		#a, b, cond = _solve_linearized_vecfit(num_basis, denom_basis, y)
		a, b, cond = minimize_2norm_varpro(num_basis, denom_basis, y, P_orth = False, method = 'ls')
		b_norm = b[0] #np.linalg.norm(b)
		b /= b_norm
		a /= b_norm
	
		# Compute the rational approximation
		Pa = np.einsum('ij,j...->i...', num_basis, a)
		Qb = denom_basis @ b
		r = np.multiply(1./Qb.reshape(-1, *([1,]*nout_dim)), Pa)

		residual_norm = np.linalg.norm( (y - r).flatten(), 2)
		delta_norm = np.linalg.norm( (r_old - r).flatten(), 2)
		b_norm = np.linalg.norm(b[1:]/b[0], np.inf)

		if residual_norm < best_fit['residual_norm']:
			best_fit['residual_norm'] = residual_norm
			best_fit['a'] = a
			best_fit['b'] = b
			best_fit['poles'] = poles

		if verbose:
			printer.print_iter(it = it, res = residual_norm, delta = delta_norm, bnorm = b_norm, cond = cond)	

		if it == maxiter - 1:
			if verbose: print("maximum iteration limit reached")
			break

		if delta_norm < ftol:
			if verbose: print("terminated due to small change in approximation")
			break

		if b_norm < btol:
			if verbose: print("terminated due to denominator being approximately constant")
			break
	
		# Update roots only if we are going to continue the iteration
		# (if we update these we change the polynomial values)
		# This is the root finding approach that Gustavsen takes in Vector Fitting
		# See Gus06: eq. 5
		poles = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(len(poles)), b[1:]))

		r_old = r		

	a = best_fit['a']
	b = best_fit['b']
	poles = best_fit['poles']

	if bonus_basis:
		bonus_poly = Polynomial(bonus_basis, a[len(poles):])
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
		num = np.einsum('ij,j...->i...', C, self._a)
		denom = self._b[0] + C @ self._b[1:]
		
		if self.bonus_poly is not None:
			num += self.bonus_poly(X)

		nout_dim = len(self._a.shape[1:])
		return np.multiply(1./denom.reshape(-1, *([1,]*nout_dim)), num)

			

class VectorFittingRationalApproximation(VectorFittingRationalFunction):
	def __init__(self, num_degree, denom_degree, *args, **kwargs):
		self.num_degree = int(num_degree)
		self.denom_degree = int(denom_degree)
		self.args = args
		self.kwargs = kwargs
	
	def fit(self, X, y):
		self._a, self._b, self.poles, self.bonus_poly = vecfit(X, y, self.num_degree, self.denom_degree, *self.args, **self.kwargs)

	
