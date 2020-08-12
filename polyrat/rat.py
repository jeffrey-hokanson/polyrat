import numpy as np
import scipy.linalg
import scipy.optimize
from .basis import *
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
	return VH.T.conj()[:,-1], s

def _minimize_1_norm(A):
	r"""
		Solve the optimization problem 
		
		min_x  || A @ x ||_1
	"""
	m, n = A.shape

	U, s, VH = np.linalg.svd(A, full_matrices = False)
	print(m,n, *U.shape)
	A_ub = np.vstack([
			np.hstack([U, -np.eye(m)]),
			np.hstack([-U, np.eye(m)])
			])
	b_ub = np.zeros(2*m)
	c = np.zeros(n + m)
	c[n:] = 1.

	lb = -np.inf*np.ones(m+n)
	ub = np.inf*np.ones(m+n)
	lb[n:] = 0
	
	bounds = [[None, None] for i in range(m+n)]
	for i in range(m): bounds[n+i][0] = 0

	res = scipy.optimize.linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = bounds )
	print(res.success, res.status, res.message)
	x = res.x[:n] 
	x = VH.conj().T @ (x/s)
	return res.x[:n], s

	# Warm-start with 2-norm solution
#	x0, s = _minimize_2_norm(A)
#	def obj(x):
#		return np.linalg.norm(A @ x, 1)
#	def jac(x):
#		return A.T @ np.sign(A @ x)
#		
#	res = scipy.optimize.minimize(obj, x0, jac = jac, method = 'CG', options = {'disp': 100})
#	#print(jac(res.x))
#	#print(res)
#	return res.x, s


def skfit_rebase(X, y, num_degree, denom_degree, maxiter = 20, verbose = True, xtol = 1e-7, history = False, denom0 = None, norm = 2):
	r""" The SK-iteration, but at each step use Vandermonde with Arnoldi to construct a new basis

	Parameters
	----------
	X: np.array (M,m)
		
	y: np.array (M,)
	
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
 
	elif np.isclose(norm, 1):
		linearized_solution = _minimize_1_norm
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
			P, RP, _ = vandermonde_arnoldi_CGS(X, num_degree, weight = 1./denom)	
			Q, RQ, _ = vandermonde_arnoldi_CGS(X, denom_degree, weight = 1./denom)	
			
			A = np.hstack([P, np.multiply(-y[:,None], Q) ])
			x, s = linearized_solution(A)
			x2, _ = _minimize_2_norm(A)
			print(x)
			print(x2)
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
			best_sol = (P, RP, Q, RP, x)
			best_res_norm = res_norm

		if history:
			hist.append({'fit': fit, 'cond':s[0]/s[-1]})

		if verbose:
			printer.print_iter(it = it, delta_fit = delta_fit, res_norm = res_norm, cond = s[0]/s[-1]) 


		if delta_fit < xtol:
			break

		denom *= np.abs(Qb)
		denom[denom == 0.] = 1.
		fit_old = fit

	if history:	
		return best_sol, hist
	else:
		return best_sol


class SKRationalApproximation(RationalApproximation):
	r"""

	Parameters
	----------
	
	"""

	def __init__(self, num_degree, denom_degree, refine = True, norm = 2, basis = None, update = True, maxiter = 20, verbose = True):

		RationalApproximation.__init__(self, num_degree, denom_degree)
		self.refine = refine
		self.norm = norm
		if self.norm != 2:
			raise NotImplementedError

		self.maxiter = int(maxiter)
		self.verbose = verbose

	def fit(self, X, y, denom0 = None):

		X = np.array(X)
		y = np.array(y)
		assert X.shape[0] == y.shape[0], "X and y do not have the same number of rows"

		if self.verbose:
			printer = IterationPrinter(it = '4d', l2_err = '21.15e', angle = '8.2e', cond = '8.2e', delta_fit = '8.2e')
			printer.print_header(it = 'iter', angle = '∡ btw sol', l2_err = '𝓁₂ err', cond = 'cond', delta_fit = 'Δ fit')

		if denom0 is None:
			denom0 = np.ones(len(y)) 



