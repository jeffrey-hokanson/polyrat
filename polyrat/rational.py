import numpy as np
import scipy.linalg
import scipy.optimize
from .basis import *
from .polynomial import *
from .skiter import *
from iterprinter import IterationPrinter
from .rational_ratio import *
from copy import deepcopy




def ratfit_scipy(y, P, Q, a0, b0, **kwargs):
	if all( [np.all(np.isreal(X)) for X in [y, P, Q, a0, b0]]):
		return ratfit_scipy_real(y, P, Q, a0, b0, **kwargs)
	else:
		return ratfit_scipy_complex(y, P, Q, a0, b0, **kwargs)

def ratfit_scipy_real(y, P, Q, a0, b0, **kwargs):
	x0 = np.hstack([a0, b0])
	m = P.shape[1]
	n = Q.shape[1]

	res = lambda x: _rational_residual_real(x, P, Q, y)
	jac = lambda x: _rational_jacobian_real(x, P, Q)
	
	# Although the Jacobian is structurally rank deficent, 
	# we count on the trust region to effectively add a regularization
	# making the Jacobian full rank	
	res = scipy.optimize.least_squares(res, x0, jac, **kwargs)
#	for key in ['nfev', 'njev', 'message']:
#		print(key, res[key])
	
	a = res.x[:m]
	b = res.x[-n:]
	return a, b


def ratfit_scipy_complex(y, P, Q, a0, b0, **kwargs):
	a0 = np.copy(a0).astype(np.complex)
	b0 = np.copy(b0).astype(np.complex)
	x0 = np.hstack([a0.view(float), b0.view(float)])
	m = P.shape[1]
	n = Q.shape[1]

	def res(x):
		a = x[:2*m].view(complex)
		b = x[-2*n:].view(complex)
		res = (P @ a)/(Q @ b) - y
		return res.view(float)

	def jac(x): 
		a = x[:2*m].view(complex)
		b = x[-2*n:].view(complex)
		Pa = P @ a
		Qb = Q @ b
		J = np.hstack([			
				np.multiply((1./Qb)[:,None], P),				
				np.multiply(-(Pa/Qb**2)[:,None], Q),
			])
		JRI = np.zeros((J.shape[0]*2, J.shape[1]*2), dtype = np.float)
		JRI[0::2,0::2] = J.real
		JRI[1::2,1::2] = J.real
		JRI[0::2,1::2] = -J.imag
		JRI[1::2,0::2] = J.imag
		return JRI
		

	res = scipy.optimize.least_squares(res, x0, jac, **kwargs)
	a = res.x[:2*m].view(complex)
	b = res.x[-2*n:].view(complex)	

	return a,b


class RationalFunction:
	pass


class RationalApproximation:
	def __init__(self, num_degree, denom_degree):
		self.num_degree = num_degree
		self.denom_degree = denom_degree


class RationalRatio(RationalFunction):
	def __init__(self, numerator, denominator):
		self.numerator = deepcopy(numerator)
		self.denominator = deepcopy(denominator)

	@property
	def P(self):
		return self.numerator.basis.basis()

	@property
	def Q(self):
		return self.denominator.basis.basis()

	@property
	def a(self):
		return self.numerator.coef

	@property
	def b(self):
		return self.denominator.coef

class SKRationalApproximation(RationalApproximation, RationalRatio):
	r"""

	Parameters
	----------
	
	"""

	def __init__(self, num_degree, denom_degree, refine = True, norm = 2, 
		Basis = None, rebase = True, maxiter = 20, verbose = True):

		RationalApproximation.__init__(self, num_degree, denom_degree)
		self.refine = refine
		self.norm = norm
		#if self.norm != 2:
		#	raise NotImplementedError

		self.maxiter = int(maxiter)
		self.verbose = verbose
		self.rebase = rebase
		if Basis is None:
			Basis = LegendrePolynomialBasis
		
		self.Basis = Basis

		self.numerator = None
		self.denominator = None

	def fit(self, X, y, denom0 = None):

		X = np.array(X)
		y = np.array(y)
		assert X.shape[0] == y.shape[0], "X and y do not have the same number of rows"

		if self.rebase:
			self.numerator, self.denominator = skfit_rebase(
				X, y, self.num_degree, self.denom_degree,
				maxiter = self.maxiter, verbose = self.verbose, norm = self.norm
				)
		else:
			num_basis = self.Basis(X, self.num_degree)	
			denom_basis = self.Basis(X, self.denom_degree)	
			P = num_basis.basis()
			Q = denom_basis.basis()
		
			a, b = skfit(y, P, Q, maxiter = self.maxiter, verbose = self.verbose, norm = self.norm)

			self.numerator = Polynomial(num_basis, a)
			self.denominator = Polynomial(denom_basis, b)

		if self.refine and False:
			# Use an optimization method to get a better result
			P = self.numerator.basis.basis()
			Q = self.denominator.basis.basis()
			a0 = self.numerator.coef
			b0 = self.denominator.coef

			a, b = ratfit_scipy(y, P, Q, a0, b0, gtol = 1e-12, ftol = 1e-10) 
			self.numerator.coef = np.copy(a)
			self.denominator.coef = np.copy(b)

			if self.verbose:
				res_norm = np.linalg.norm( (P @ a)/(Q @ b) - y, self.norm)
				print(f"final residual norm {res_norm:21.15e}")
				

	def __call__(self, X):
		p = self.numerator(X)
		q = self.denominator(X)
		return p/q	
					

