r""" A stabilized variant of the Sanathanan-Koerner iteration

"""

import numpy as np
from numpy.linalg import LinAlgError
from iterprinter import IterationPrinter
from .skiter import _minimize_2_norm, _minimize_inf_norm
from .arnoldi import *
from .rational import *

from .util import minimize_2norm_varpro 



def skfit_stabilized(X, y, num_degree, denom_degree, maxiter = 20, verbose = True, 
	xtol = 1e-7, history = False, denom0 = None, norm = 2):
	r""" The Stabilized Sanathanan-Koerner Iteration


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
	elif ~np.isfinite(norm):
		linearized_solution = _minimize_inf_norm
	else: 
		raise NotImplementedError
	
	if verbose:
		printer = IterationPrinter(it = '4d', res_norm = '21.15e', delta_fit = '8.2e', cond = '8.2e')
		printer.print_header(it = 'iter', res_norm = 'residual norm', delta_fit = 'delta fit', cond = 'cond')

	# As there are no guarntees about convergence,
	# we record the best iteration
	best_res_norm = np.inf
	best_sol = None	
	nout_dim = len(y.shape[1:])
	
	# For comparison with current iterate to determine termination
	fit_old = np.zeros(y.shape, dtype = X.dtype)

	for it in range(maxiter):
		try:
			num_basis = ArnoldiPolynomialBasis(X, num_degree, weight = 1./denom)
			denom_basis = ArnoldiPolynomialBasis(X, denom_degree, weight = 1./denom)
			P = num_basis.vandermonde_X
			Q = denom_basis.vandermonde_X
			
			if np.isclose(norm, 2):
				a, b, cond = minimize_2norm_varpro(P, Q, y)
			else:		
				A = np.hstack([P, np.multiply(-y[:,None], Q) ])
				x, cond = linearized_solution(A)
				a = x[:P.shape[1]]
				b = x[-Q.shape[1]:]
	
			Pa = np.einsum('ij,j...->i...', P, a)
			Qb = Q @ b

			#fit = Pa/Qb
			fit = np.multiply(1./Qb.reshape(-1, *([1,]*nout_dim)), Pa)

			delta_fit = np.linalg.norm( (fit - fit_old).flatten(), norm)		
			res_norm = np.linalg.norm( (fit - y).flatten(), norm)
		
		except (LinAlgError) as e:
			if verbose: print(e)
			break
	

		# If we have improved the fit, append this 
		if res_norm < best_res_norm:
			numerator = Polynomial(num_basis, a)
			denominator = Polynomial(denom_basis, b)
			best_sol = [numerator, denominator]
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
		return best_sol + [hist]
	else:
		return best_sol



class StabilizedSKRationalApproximation(RationalApproximation, RationalRatio):
	r"""

	Parameters
	----------
	
	"""

	def __init__(self, num_degree, denom_degree, norm = 2, maxiter = 20, verbose = True, xtol = 1e-7):

		RationalApproximation.__init__(self, num_degree, denom_degree)
		self.norm = norm
		self.xtol = float(xtol)

		self.maxiter = int(maxiter)
		self.verbose = verbose
		
		self.numerator = None
		self.denominator = None

	def fit(self, X, y, denom0 = None):
		X = np.array(X)
		y = np.array(y)
		assert X.shape[0] == y.shape[0], "X and y do not have the same number of rows"

		self.numerator, self.denominator, self.hist = skfit_stabilized(
			X, y, self.num_degree, self.denom_degree,
			maxiter = self.maxiter, verbose = self.verbose, norm = self.norm,
			history = True, xtol = self.xtol, denom0 = denom0,
			)
		
