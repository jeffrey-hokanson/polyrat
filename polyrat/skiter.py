""" Routines implementing variants of the Sanathanan-Koerner iteration
"""
import numpy as np
from numpy.linalg import LinAlgError
import scipy.linalg
import scipy.optimize
from .basis import *
from .arnoldi import *
from .polynomial import *
from .rational import *
from .util import minimize_2norm_dense, minimize_2norm_varpro 
from iterprinter import IterationPrinter


def _minimize_2_norm(A):
	r"""
		Solve the optimization problem 
		
		min_x  || A @ x ||_2
	"""
	U, s, VH = scipy.linalg.svd(A, full_matrices = False)
		
	# Condition number of singular vectors, cf. Stewart 01: Eq. 3.16
	with np.errstate(divide = 'ignore'):
		cond = s[0]*np.sqrt(2)/(s[-2] - s[-1])
	return VH.T.conj()[:,-1], cond


def _minimize_inf_norm(A, nsamp = 360):
	r"""

	min_x || A @ x ||_inf s.t. ||x||_2 = 1

	If A is real, then x is real

	"""

	if np.all(np.isreal(A)):
		return _minimize_inf_norm_real(A)
	else:
		return _minimize_inf_norm_complex(A, nsamp = nsamp)


def _minimize_inf_norm_real(A):
	m,n = A.shape
	A_ub = np.vstack([ 
		# A @ x - t <= 0
		np.hstack([A, -np.ones((m,1))]),
		# -A @ x - t <= 0
		np.hstack([-A, -np.ones((m,1))]),
		])
	b_ub = np.zeros(2*m)
	
	c = np.zeros(n + 1)
	c[-1] = 1.
	bounds = np.zeros((n+1, 2))
	bounds[:n,0] = -np.inf
	bounds[:,1] = np.inf
	
	x, cond = _minimize_2_norm(A) 
	A_eq = np.zeros((1,n+1))
	A_eq[0,:n] = x.real
	b_eq = np.ones((1,))

	res = scipy.optimize.linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds,
		options = {'cholesky': False, 'sym_pos':False, 'lstsq': True})
	x = res.x[:-1]
	return x/np.linalg.norm(x), cond

def _minimize_inf_norm_complex(A, nsamp = 360):
	m, n = A.shape
	A_ub = [
		# Ar @ xr - Ai @ xi - tr <=0 
		np.hstack([A.real, -A.imag, -np.ones((m, 1)), np.zeros((m, 1)), np.zeros((m,1)) ]),
		# Ai @ xr + Ar @ xi - ti <=0 
		np.hstack([A.real, A.imag, np.zeros((m, 1)), -np.ones((m, 1)), np.zeros((m,1)) ]),
		# -Ar @ xr + Ai @ xi - tr <=0 
		np.hstack([-A.real, A.imag, -np.ones((m, 1)), np.zeros((m, 1)), np.zeros((m,1)) ]),
		# -Ai @ xr - Ar @ xi - ti <=0 
		np.hstack([-A.real, -A.imag, np.zeros((m, 1)), -np.ones((m, 1)), np.zeros((m,1)) ]),
		]
	# constraints on tr + ti
	th = np.linspace(0, 2*np.pi, nsamp, endpoint = False)
	A_ub.append(
		np.hstack( [np.zeros((nsamp, 2*n)), np.cos(th).reshape(-1,1), np.sin(th).reshape(-1,1), -np.ones((nsamp,1))])
		)
	A_ub = np.vstack(A_ub)

	b_ub = np.zeros(A_ub.shape[0])
	
	c = np.zeros(2*n + 3)
	c[-1] = 1.
	
	bounds = np.zeros((2*n+3, 2))
	bounds[:2*n,0] = -np.inf
	bounds[:,1] = np.inf


	x, cond = _minimize_2_norm(A) 
	A_eq = np.hstack([x.real, x.imag, np.zeros(3)]).reshape(1,-1)
	b_eq = np.ones((1,))

	res = scipy.optimize.linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds,
		options = {'cholesky': False, 'sym_pos':False, 'lstsq': True})
	
	x = res.x[:n] + 1j*res.x[n:2*n]
	
	return x/np.linalg.norm(x), cond


def _minimize_1_norm(A):
	raise NotImplementedError


#def _minimize_1_norm_real(A):
#	r"""
#		Solve the optimization problem 
#		
#		min_x  || A @ x ||_1  s.t.  ||x||_2 = 1
#	"""
#	m, n = A.shape
#
#	U, s, VH = np.linalg.svd(A, full_matrices = False)
#	print(m,n, *U.shape)
#	A_ub = np.vstack([
#			# A x - t <= 0
#			np.hstack([A, -np.eye(m)]),
#			# -A x - t <= 0 
#			np.hstack([-A, -np.eye(m)])
#			])
#	b_ub = np.zeros(2*m)
#
#	# Pin one of the variables so we have a non-zero solution
#	A_eq = np.zeros((1, m +n))
#	A_eq[0,0] = 1
#	b_eq = np.ones((1,))
#
#	# Objective: minimize the sum of the upper bounds
#	c = np.zeros(n + m)
#	c[n:] = 1.
#
#	lb = -np.inf*np.ones(m+n)
#	ub = np.inf*np.ones(m+n)
#	lb[n:] = 0
#	
#	bounds = [[None, None] for i in range(m+n)]
#	for i in range(m): bounds[n+i][0] = 0
#
#	res = scipy.optimize.linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds,
#			options = {'presolve': True, 'autoscale': True, 'lstsq': True})
#	
#	#y = res.x[:n]
#	# U @ y = A @ x 
#	# U @ y = U @ np.diag(s) @ VH @ x
#	# y = np.diag(s) @ VH @ x
#	#x = (VH.conj().T @ (y/s ))
#	x = res.x[:n]
#	x /= np.linalg.norm(x)
#	x *= np.sign(x[0])
#	return x, s
		


def skfit(y, P, Q, maxiter = 20, verbose = True, history = False, denom0 = None, norm = 2, xtol = 1e-7):
	r"""


	Returns
	-------
	a: np.array 
		Numerator coefficients
	b: np.array
		Denominator coefficients
	"""	
	if denom0 is None:
		denom = np.ones(len(y), dtype = y.dtype)
	else:
		assert len(denom0) == len(y)
		denom = np.array(denom0)
	
	if np.isclose(norm, 2):
		linearized_solution = _minimize_2_norm
	elif ~np.isfinite(norm):
		linearized_solution = _minimize_inf_norm
	else: 
		raise NotImplementedError

	if verbose:
		printer = IterationPrinter(it = '4d', res_norm = '21.15e', delta_fit = '8.2e', cond = '8.2e')
		printer.print_header(it = 'iter', res_norm = 'residual norm', delta_fit = 'delta fit', cond = 'cond')
	
	if history:
		hist = []

	# As there are no guarntees about convergence,
	# we record the best iteration
	best_res_norm = np.inf
	best_sol = None	
	
	# For comparison with current iterate to determine termination
	fit_old = np.zeros(y.shape, dtype = y.dtype)

	nout_dim = len(y.shape[1:])
	
	for it in range(maxiter):
		if np.isclose(norm, 2):
			dP = np.multiply((1./denom)[:,None], P)
			dQ = np.multiply((1./denom)[:,None], Q)

			a, b, cond = minimize_2norm_varpro(dP, dQ, y)
			
			Pa = np.einsum('ij,j...->i...', P, a)
			Qb = Q @ b
			fit = np.multiply(1./Qb.reshape(-1, *([1,]*nout_dim)), Pa)
		else:
			A = np.hstack([ 
					np.multiply((1./denom)[:,None], P), 	
					np.multiply((-y/denom)[:,None], Q)
				])

			x, cond = linearized_solution(A)
			
			a = x[:P.shape[1]]
			b = x[-Q.shape[1]:]

			Pa = P @ a
			Qb = Q @ b
			fit = Pa/Qb

		res_norm = np.linalg.norm( (y - fit).flatten(), norm)

		if res_norm < best_res_norm:
			best_res_norm = res_norm
			best_sol = [a, b]

		delta_fit = np.linalg.norm( (fit - fit_old).flatten(), norm)		
		
		if history:
			hist.append({'fit': fit, 'cond': cond})

		if verbose:
			printer.print_iter(it = it, delta_fit = delta_fit, res_norm = res_norm, cond = cond) 

		if delta_fit < xtol:
			break

		# Copy over data for next loop
		denom = Qb 
		fit_old = fit

	if history:	
		return best_sol + [hist]
	else:
		return best_sol


class SKRationalApproximation(RationalApproximation, RationalRatio):
	r"""

	Parameters
	----------
	
	"""

	def __init__(self, num_degree, denom_degree, norm = 2, 
		Basis = None, maxiter = 20, verbose = True, xtol = 1e-7):

		RationalApproximation.__init__(self, num_degree, denom_degree)
		self.norm = norm
		self.xtol = float(xtol)
		#if self.norm != 2:
		#	raise NotImplementedError

		self.maxiter = int(maxiter)
		self.verbose = verbose
		if Basis is None:
			Basis = LegendrePolynomialBasis
		
		self.Basis = Basis

		self.numerator = None
		self.denominator = None

	def fit(self, X, y, denom0 = None):
		X = np.array(X)
		y = np.array(y)
		assert X.shape[0] == y.shape[0], "X and y do not have the same number of rows"

		num_basis = self.Basis(X, self.num_degree)	
		denom_basis = self.Basis(X, self.denom_degree)	
		P = num_basis.vandermonde_X
		Q = denom_basis.vandermonde_X
	
		a, b, self.hist = skfit(y, P, Q, maxiter = self.maxiter, verbose = self.verbose, norm = self.norm, history = True, 
			xtol = self.xtol, denom0 = denom0)

		self.numerator = Polynomial(num_basis, a)
		self.denominator = Polynomial(denom_basis, b)
		

