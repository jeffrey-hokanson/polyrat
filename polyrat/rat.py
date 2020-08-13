import numpy as np
import scipy.linalg
import scipy.optimize
from .basis import *
from .poly import *
from iterprinter import IterationPrinter



class RationalApproximation:
	def __init__(self, num_degree, denom_degree):
		self.num_degree = num_degree
		self.denom_degree = denom_degree
		

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


def _minimize_inf_norm(A):
	pass


#def _minimize_1_norm(A):
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
	fit_old = np.zeros(y.shape[0], dtype = y.dtype)

	
	for it in range(maxiter):
		A = np.hstack([ 
				np.multiply((1./denom)[:,None], P), 	
				np.multiply((-y/denom)[:,None], Q)
			])
		x, cond = _minimize_2_norm(A)
		
		a = x[:P.shape[1]]
		b = x[-Q.shape[1]:]
		Pa = P @ a
		Qb = Q @ b
		fit = Pa/Qb
		res_norm = np.linalg.norm(y - fit, norm)

		if res_norm < best_res_norm:
			best_res_norm = res_norm
			best_sol = (a, b)

		delta_fit = np.linalg.norm(fit - fit_old, norm)		
		
		if history:
			hist.append({'fit': fit, 's': s})

		if verbose:
			printer.print_iter(it = it, delta_fit = delta_fit, res_norm = res_norm, cond = cond) 

		if delta_fit < xtol:
			break

		# Copy over data for next loop
		denom = Qb 
		fit_old = fit

	if history:	
		return best_sol, hist
	else:
		return best_sol



def skfit_rebase(X, y, num_degree, denom_degree, maxiter = 20, verbose = True, 
	xtol = 1e-7, history = False, denom0 = None, norm = 2):
	r""" The SK-iteration, but at each step use Vandermonde with Arnoldi to construct a new basis

	Parameters
	----------
	X: np.array (M,dim)
		
	y: np.array (M,)


	Returns
	-------
			
	"""

	if history:
		hist = []


	if denom0 is None:
		denom = np.ones(X.shape[0], dtype = X.dtype)
	else:
		assert denom0.shape[0] == X.shape[0]
		denom = denom0


	if np.isclose(norm, 2):
		linearized_solution = _minimize_2_norm
	else: 
		raise NotImplementedError
	
	if verbose:
		printer = IterationPrinter(it = '4d', res_norm = '21.15e', delta_fit = '8.2e', cond = '8.2e')
		printer.print_header(it = 'iter', res_norm = 'residual norm', delta_fit = 'delta fit', cond = 'cond')

	# As there are no guarntees about convergence,
	# we record the best iteration
	best_res_norm = np.inf
	best_sol = None	
	
	# For comparison with current iterate to determine termination
	fit_old = np.zeros(y.shape[0], dtype = X.dtype)

	for it in range(maxiter):
		try:
			num_basis = ArnoldiPolynomialBasis(X, num_degree, weight = 1./denom)
			denom_basis = ArnoldiPolynomialBasis(X, denom_degree, weight = 1./denom)
			P = num_basis.basis()
			Q = denom_basis.basis()	
			#P, RP, _ = vandermonde_arnoldi_CGS(X, num_degree, weight = 1./denom)	
			#Q, RQ, _ = vandermonde_arnoldi_CGS(X, denom_degree, weight = 1./denom)	
			
			A = np.hstack([P, np.multiply(-y[:,None], Q) ])
			x, cond = linearized_solution(A)
			a = x[:P.shape[1]]
			b = x[-Q.shape[1]:]
			
			Pa = P @ a
			Qb = Q @ b

			fit = Pa/Qb

			delta_fit = np.linalg.norm(fit - fit_old, norm)		
			res_norm = np.linalg.norm(fit - y, norm)

		except ValueError as e:
			print(e)
			break
	

		# If we have improved the fit, append this 
		if res_norm < best_res_norm:
			numerator = Polynomial(num_basis, a)
			denominator = Polynomial(denom_basis, b)
			best_sol = (numerator, denominator)
			best_res_norm = res_norm

		if history:
			hist.append({'fit': fit, 'cond':cond})

		if verbose:
			printer.print_iter(it = it, delta_fit = delta_fit, res_norm = res_norm, cond = cond) 

		if delta_fit < xtol:
			break

		denom = np.abs(denom * Qb)
		denom[denom == 0.] = 1.
		fit_old = fit

	if history:	
		return best_sol, hist
	else:
		return best_sol


################################################################################
# Refinement algorithms a rational fit based on optimization critera
################################################################################

def ratfit_scipy(y, P, Q, a0, b0, **kwargs):
	if all( [np.all(np.isreal(X)) for X in [y, P, Q, a0, b0]]):
		return ratfit_scipy_real(y, P, Q, a0, b0, **kwargs)
	else:
		return ratfit_scipy_complex(y, P, Q, a0, b0, **kwargs)

def ratfit_scipy_real(y, P, Q, a0, b0, **kwargs):
	x0 = np.hstack([a0, b0])
	m = P.shape[1]
	n = Q.shape[1]

	def res(x):
		Pa = P @ x[:m]
		Qb = Q @ x[-n:]
		return Pa/Qb - y 

	def jac(x):
		Pa = P @ x[:m]
		Qb = Q @ x[-n:]
		return np.hstack([			
				np.multiply((1./Qb)[:,None], P),				
				np.multiply(-(Pa/Qb**2)[:,None], Q),
			])
	
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


class SKRationalApproximation(RationalApproximation):
	r"""

	Parameters
	----------
	
	"""

	def __init__(self, num_degree, denom_degree, refine = True, norm = 2, 
		Basis = None, rebase = True, maxiter = 20, verbose = True):

		RationalApproximation.__init__(self, num_degree, denom_degree)
		self.refine = refine
		self.norm = norm
		if self.norm != 2:
			raise NotImplementedError

		self.maxiter = int(maxiter)
		self.verbose = verbose
		self.rebase = rebase
		if Basis is None:
			Basis = LegendrePolynomialBasis
		
		self.Basis = Basis

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

		if self.refine:
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
					

