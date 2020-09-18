r""" Parametric-AAA

"""

from itertools import product
import numpy as np
from copy import deepcopy as copy
import scipy.linalg
from iterprinter import IterationPrinter

from .rational import RationalBarycentric

def _build_cauchy(x,y):
	with np.errstate(divide = 'ignore', invalid = 'ignore'):
		C = 1./(np.tile(x.reshape(-1,1), (1,len(y))) - np.tile(y.reshape(1,-1), (len(x),1)))
		# If we have a divide by zero error, we reason that for that row,
		# only the matching column is non-zero
		for row in np.argwhere(~np.all(np.isfinite(C), axis = 1)):
			C[row] = np.zeros(C.shape[1])
			C[row, np.argmin(np.abs(x[row] - y)).flatten()] = 1
	return C 

def eval_paaa(Xe, X, y, I, b, basis, order):
	d = X.shape[1]

	C = np.ones((len(Xe),1), dtype = X.dtype)
	for i in range(d):
		Ct = _build_cauchy(Xe[:, i], basis[i])
		C = np.vstack([C[:,j] * Ct[:,k] for j, k in product(range(C.shape[1]), range(Ct.shape[1]))]).T

	num = np.einsum('ij,j...->i...', C, [bi*yi for bi, yi in zip(b,y[order])] )
	denom = np.einsum('ij,j...->i...', C, b)
	
	with np.errstate(divide = 'ignore', invalid = 'ignore'):
		return np.einsum('i...,i->i...', num, 1./denom)

def paaa(X, y, verbose = True, maxiter = 100, tol = None):
	r"""

	"""
	X = np.array(X)
	y = np.array(y)
	assert X.shape[0] == y.shape[0], "X and y must have the same first dimension"

	if tol is None:
		tol = 1e-10

	d = X.shape[1]
	r = np.zeros(y.shape)

	degree = np.zeros(d, dtype = np.int)


	if verbose:
		printer = IterationPrinter(it = '4d', degree = f'{2+3*d:d}s', norm = '20.10e', cond = '10.2e')
		printer.print_header(it = 'iter', degree = 'degree', norm = 'norm mismatch', cond = 'cond #')
		printer.print_iter(norm = np.linalg.norm(y))
	# indices of points where we interpolate
	I = np.zeros(y.shape[0], dtype = np.bool)

	
	mismatch = np.copy(y)
	best = None
	for it in range(maxiter):
		# compute the pointwise-residual
		residual = np.sum(np.abs(mismatch), axis = tuple(range(1,len(y[0].shape)+1)))
		# zero out points we've already identified so we don't re-evaluate
		residual[I] = 0
		
		# find the point with the largest residual and add that to the interpolation set
		i = np.argmax(residual)
		I[i] = True
		
		# determine points in tensor-product barycentric basis
		basis = [np.array(list(set(X[I,i]))).reshape(-1) for i in range(d)]
		degree = np.array([len(b)-1 for b in basis], dtype = np.int)

		# With this algorithm we need to "complete the square"
		# by adding all terms of the form
		I = np.all( [np.any([X[:,i] == bi for bi in basis[i]], axis = 0) for i in range(d)], axis = 0)
		#print("I", np.argwhere(I).flatten())
		assert np.prod([len(b) for b in basis]) == np.sum(I), "Missing points in tensor product grid"
		
		# Build the multivariate Cauchy matrix
		C = np.ones((np.sum(~I),1), dtype = X.dtype)
		for i in range(d):
			Ct = _build_cauchy(X[~I, i], basis[i])
			C = np.vstack([C[:,j] * Ct[:,k] for j, k in product(range(C.shape[1]), range(Ct.shape[1]))]).T

		# Build the Loewner matrix associated with each input
		Lten = []
		# The issue is we need to match the ordering of I to that in C
		order = [ int(np.argwhere( (X == xx).all(axis=1))) for xx in product(*basis)]
		for idx in np.ndindex(y[0].shape):
			Lten.append( (C.T * y[(~I,*idx)]).T - C*y[(order,*idx)] )
		L = np.vstack(Lten)
	
		# Compute coefficients for denominator polynomial
		U, s, VH = scipy.linalg.svd(L, full_matrices = False, compute_uv = True, overwrite_a = True)
		b = VH.conj().T[:,-1] 
		
		if len(s) >= 2:
			with np.errstate(divide='ignore',invalid='ignore'):
				cond = s[0]*np.sqrt(2)/(s[-2] - s[-1])
		else:
			cond = None
	
		r = eval_paaa(X, X, y, I, b, basis, order)
		if np.all(np.isfinite(r)):
			best = (np.copy(I), np.copy(b), copy(basis), copy(order))
			assert np.all(np.isclose(r[I], y[I])), "Constructed rational approximant failed to correctly interpolate"
		else:
			if verbose:
				print("Encountered nan in evaluation")
			break
			
		mismatch = y - r

		mismatch_norm = np.linalg.norm(mismatch)

		if verbose:
			degree_str = '(' + ','.join([f'{di:2d}' for di in degree]) + ')'
			printer.print_iter(it = it, norm = mismatch_norm, degree = degree_str, cond = cond) 

		if mismatch_norm < tol:
			if verbose: print('terminated due to small mismatch')
			break
	
	return best


class ParametricAAARationalApproximation(RationalBarycentric):
	def __init__(self, tol = None, verbose = True, maxiter = 100):
		self.tol = None
		self.verbose = verbose
		self.maxiter = maxiter
		self.degree = None

	def fit(self, X, y):
		self.X = np.copy(X)
		self.y = np.copy(y)
		self._I, self._b, self._basis, self._order = paaa(X, y, verbose = self.verbose, maxiter = self.maxiter)
		self.degree = np.array([len(bi)-1 for bi in self._basis], dtype = np.int)	


	def __call__(self, X):
		return eval_paaa(X, self.X, self.y, self._I, self._b, self._basis, self._order)

	@property
	def interpolation_points(self):
		return np.array(list(product(*self._basis)))
		
