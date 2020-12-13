r""" Code for AAA rational approximation
"""

import numpy as np
import scipy.linalg
from iterprinter import IterationPrinter
from .rational import RationalBarycentric

#def eval_barcentric(xeval, x, y, I, a, b):
#	"""
#	Parameters
#	----------
#	xeval: np.array
#		Locations to evaluate the function
#	x: np.array
#		input coordinates for fit
#	"""
#
#	with np.errstate(divide='ignore',invalid='ignore'):
#		A = np.hstack([ 1. /(xeval - xi) for xi in zip(x[I])])
#		# For the rows that have a divide by zero error, 
#		# we replace with value corresponding to the limit
#		for row in np.argwhere(~np.all(np.isfinite(A), axis = 1)):
#			A[row] = np.zeros(A.shape[1])
#			A[row, np.argmin(np.abs(xeval[row] - x[I])).flatten()] = 1 
#		
#	num = np.einsum('ij,j...->i...', A, a)
#	denom = np.einsum('ij,j->i', A,b)
#	reval = np.einsum('i...,i->i...', num, 1./denom)
#	
#	return reval	



def eval_aaa(xeval, x, y, I, b):
	""" Evaluate an AAA rational approximation
	Parameters
	----------
	xeval: np.array
		Locations to evaluate the function
	x: np.array
		input coordinates for fit
	"""
	xeval = xeval.reshape(-1,1)
	with np.errstate(divide='ignore',invalid='ignore'):
		A = np.hstack([ bi /(xeval - xi) for bi, xi in zip(b, x[I])])
		# For the rows that have a divide by zero error, 
		# we replace with value corresponding to the limit
		for row in np.argwhere(~np.all(np.isfinite(A), axis = 1)):
			A[row] = np.zeros(A.shape[1])
			A[row, np.argmin(np.abs(xeval[row] - x[I])).flatten()] = 1 
		
	num = np.einsum('ij,j...->i...', A, y[I])
	denom = np.sum(A, axis = 1)
	reval = np.einsum('i...,i->i...', num, 1./denom)
	
	return reval	


def _build_cauchy(x,y):
	return 1./(np.tile(x.reshape(-1,1), (1,len(y))) - np.tile(y.reshape(1,-1), (len(x),1)))

def aaa(x, y, degree = None, tol = None, verbose = True):
	r""" A vector-valued Adaptive Anderson-Antoulas implementation

	Parameters
	----------
	x: np.array (M,)
		input coordinates
	y: np.array (M,...)
		output coordinates
	"""

	assert not ((degree is None) and (tol is None)), "One or both of 'degree' and 'tol' must be specified" 

	if degree is None:
		degree = len(x)//2 - 1


	mismatch = y
	I = np.zeros(len(y), dtype = np.bool)	# Index of x values used as barycentric nodes 

	if verbose:
		printer = IterationPrinter(it = '4d', res = '20.16e', cond = '8.3e')
		printer.print_header(it = 'iter', res = 'Frobenius norm residual', cond = 'condition number')

	for it in range(degree+1):
		# TODO: Is sum best? or how about maximum?
		residual = np.sum(np.abs(mismatch), axis = tuple(range(1,len(y[0].shape)+1)))
		residual[I] = 0 	# zero out residual at nodes that have already been selected
		
		Inew = np.argmax(residual)
		I[Inew] = True

		# Construct the Cauchy matrix 
		C = _build_cauchy(x[~I], x[I])
	 
		# Build the Loewner matrix associated with each input
		Lten = []
		for idx in np.ndindex(y[0].shape):
			Lten.append( (C.T * y[(~I,*idx)]).T - C*y[(I,*idx)] )
		L = np.vstack(Lten)
			
		# Compute coefficients for denominator polynomial
		U, s, VH = scipy.linalg.svd(L, full_matrices = False, compute_uv = True, overwrite_a = True)
		b = VH.conj().T[:,-1] 
		if len(s) >= 2:
			with np.errstate(divide='ignore',invalid='ignore'):
				cond = s[0]*np.sqrt(2)/(s[-2] - s[-1])
		else:
			cond = None

		mismatch = y - eval_aaa(x, x, y, I, b) 	
		res_norm = np.sqrt(np.sum(np.abs(mismatch)**2))
	
		if verbose:
			printer.print_iter(it=it, res = res_norm, cond = cond)	

		if tol is not None:
			if tol > res_norm:
				if verbose: print("terminated due to small residual")
				break

	return I, b 	

class AAARationalApproximation(RationalBarycentric):
	
	def __init__(self, degree = None, tol = None, verbose = True):
		self.degree = degree
		self.tol = tol
		self.verbose = verbose

		if self.degree is None and self.tol is None:
			self.tol = 1e-12

	def fit(self, X, y):
		X = np.array(X)
		self.y = np.array(y)
		assert len(X) == len(y), "Length of X and y do not match"
		self.x = X.flatten()
		assert len(self.x) == len(y), "AAA only supports scalar-valued inputs"

		self.I, self.b = aaa(self.x, self.y, degree = self.degree, tol = self.tol, verbose = self.verbose)

	def __call__(self, X):
		x = np.array(X).flatten().reshape(-1,1)
		assert len(x) == len(X), "X must be a scalar-valued input"
		return eval_aaa(x, self.x, self.y, self.I, self.b)
		

class AAALawsonRationalApproximation(RationalBarycentric):
	pass



