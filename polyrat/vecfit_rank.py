r""" Rank-constrained vector fitting
"""

from itertools import product

import numpy as np
import scipy.linalg
from .aaa import _build_cauchy
from .linratfit import LinearizedRationalApproximation 
from .vecfit import VectorFittingRationalApproximation
from .util import _zeros

from iterprinter import IterationPrinter


def _fit_fgh(Y, C, f, g):
	# For numerical experiments only; slow
	import cvxpy as cp
	M, p, m = Y.shape
	r = C.shape[1]
	
	df = cp.Variable((r, p), complex = True)
	dg = cp.Variable((r, m), complex = True)
	h = cp.Variable((r,), complex = True)

	obj = 0
	Hr = [0 for i in range(C.shape[0])]
	for i in range(M):
		for k in range(r):
			#Hrk = f[k:k+1].T @ g[k:k+1].conj() # 0th order
			Hrk = f[k:k+1].T @ cp.conj(dg[k:k+1])
			Hrk += df[k:k+1].T @ g[k:k+1].conj()
			Hr[i] += Hrk*C[i,k]
		obj += cp.sum_squares(Y[i]*(1 + C[i,:] @ h) - Hr[i])
	
	prob = cp.Problem(cp.Minimize(obj))
	prob.solve(verbose = False, solver = 'OSQP',eps_abs = 1e-8, eps_rel = 1e-8)
	#print('df', df.value)
	#print('dg', dg.value)
	fnew = df.value
	gnew = dg.value
	return fnew, gnew, h.value

def _fit_f(Y, C, g):
	r""" Find the left-hand side vectors only

	"""
	M, p, m = Y.shape
	r = C.shape[1]
	
	# Allocate storage
	A = _zeros((M*m, r), Y, C, g)
	f = _zeros((r, p), Y, C, g)

	for j in range(p):
		for k in range(m):
			A[k*M:(k+1)*M] = np.multiply(g[:,k].conj(), C)
			
		f[:,j] = scipy.linalg.lstsq(A, Y[:,j].T.flatten(), 
			overwrite_a = True, overwrite_b = False, check_finite = False, 
			lapack_driver = 'gelsy')[0]

	return f


def _fit_g(Y, C, f):
	M, p, m = Y.shape
	r = C.shape[1]
	
	# Allocate storage
	A = _zeros((M*p, r), Y, C, f)
	g = _zeros((r, m), Y, C, f)

	for j in range(m):
		for k in range(p):
			A[k*M:(k+1)*M] = np.multiply(f[:,k], C)
	
		g[:,j] = scipy.linalg.lstsq(A, Y[:,:,j].T.flatten(), 
			overwrite_a = True, overwrite_b = False, check_finite = False, 
			lapack_driver = 'gelsy')[0]

	return g.conj()


def _fit_h(Y, C, f, g):
	# NOTE: This function has not yet been optimized
	M, p, m = Y.shape
	r = C.shape[1]

	Ymis = np.copy(Y.flatten())
	for k in range(r):
		Ymis -= np.kron(C[:,k], np.outer(f[k], g[k].conj()).flatten())

	A = Y.flatten()[:, None] * np.kron(C, np.ones((m*p,1)))
	b = -Ymis
	x, res, rank, s = scipy.linalg.lstsq(A, b, overwrite_a = True, overwrite_b = False)	
	return x


def _fit_fh(Y, C, g):
	M, p, m = Y.shape
	r = C.shape[1]

	Q = _zeros((M*m, r, p), Y, C, g)
	R = _zeros((r, r, p), Y, C, g)
	A = _zeros((M*p*m, r), Y, C, g)
	b = _zeros((M*p*m), Y, C, g)
	f = _zeros((r, p), Y, C, g)

	Atmp = _zeros((M*m, r), Y, C, g)

	for s in range(p):
		for t in range(m):
			#Atmp[t*M:(t+1)*M,:] = C @ np.diag(g[:,t].conj())
			Atmp[t*M:(t+1)*M,:] = np.multiply(C, g[:,t].conj())

			rows = slice(s*m*M + t*M, s*m*M + (t+1)*M)
			A[rows,:] = np.diag(Y[:,s,t]) @ C
			b[rows] = Y[:,s,t]

		Q[:,:,s], R[:,:,s] = scipy.linalg.qr(Atmp, overwrite_a = True, mode = 'economic')

		# now apply the projector to both sides		
		rows = slice(s*m*M, (s+1)*m*M)
		
		b[rows] -= Q[:,:,s] @ (Q[:,:,s].T.conj() @ b[rows])
		A[rows] -= Q[:,:,s] @ (Q[:,:,s].T.conj() @ A[rows])

	# Now solve for h
	h = -scipy.linalg.lstsq(A, b)[0]

	# Now compute f
	rhs = _zeros((M*m), Y, C, g)
	Ch = C @ h
	for s in range(p):
		for t in range(m):
			# Compute: rhs[t*M:(t+1)*M] = np.diag(Y[:,s,t]) @ C @ h + Y[:,s,t]
			# Below is a compact expression using only vector operations
			rhs[t*M:(t+1)*M] = Y[:,s,t] * (Ch + 1)

		f[:,s] = scipy.linalg.solve_triangular(R[:,:,s], Q[:,:,s].T.conj() @ rhs)

	return f, h
		 


def _fit_gh(Y, C, f):
	M, p, m = Y.shape
	r = C.shape[1]

	Q = _zeros((M*p, r, m), Y, C, f)
	R = _zeros((r, r, m), Y, C, f)
	A = _zeros((M*p*m, r), Y, C, f)
	b = _zeros((M*p*m), Y, C, f)
	g = _zeros((r, m), Y, C, f)

	Atmp = _zeros((M*p, r), Y, C, f)
	
	for t in range(m):
		for s in range(p):
			#Atmp[s*M:(s+1)*M,:] = C @ np.diag(f[:,s])
			Atmp[s*M:(s+1)*M,:] = np.multiply(C, f[:,s])
			#print("ERROR", np.linalg.norm(Atmp[t*M:(t+1)*M] - np.multiply(C, f[:,s])))
			rows = slice(t*p*M + s*M, t*p*M + (s+1)*M)
			#A[rows,:] = np.diag(Y[:,s,t]) @ C
			A[rows,:] = (Y[:,s,t].reshape(-1,1)) * C
			b[rows] = Y[:,s,t]

		Q[:,:,t], R[:,:,t] = scipy.linalg.qr(Atmp, overwrite_a = True, mode = 'economic')
		
		# now apply the projector to both sides		
		rows = slice(t*p*M, (t+1)*p*M)
	
		# TODO: the conjugation here (and below) takes up a signifant running time	
		b[rows] -= Q[:,:,t] @ (Q[:,:,t].T.conj() @ b[rows])
		A[rows] -= Q[:,:,t] @ (Q[:,:,t].T.conj() @ A[rows])
	
	# Now solve for h
	h = -scipy.linalg.lstsq(A, b)[0]

	# Now compute g
	rhs = _zeros((M*p), Y, C, f)
	Ch = C @ h
	for t in range(m):
		for s in range(p):
			# Compute: rhs[t*M:(t+1)*M] = np.diag(Y[:,s,t]) @ C @ h + Y[:,s,t]
			# Below is a compact expression using only vector operations
			rhs[s*M:(s+1)*M] = Y[:,s,t] * (Ch + 1)

		g[:,t] = scipy.linalg.solve_triangular(R[:,:,t], Q[:,:,t].T.conj() @ rhs)

	return g.conj(), h

	


def eval_vecfit_rank(X, poles, f, g, h):
	Y = np.zeros((X.shape[0], f.shape[1], g.shape[1]), dtype = np.complex)

	denom = np.ones(X.shape[0], dtype = np.complex)
	for k in range(len(poles)):
		Y += np.einsum('i,jk->ijk', 1./(X[:,0] - poles[k]), np.outer(f[k], g[k].conj()))
		denom += h[k]/(X[:,0] - poles[k])

	Y *= 1./denom[:,None, None]

	return Y	



def vecfit_rank(X, Y, num_degree, denom_degree, 
	verbose = True, maxiter = 500, ftol = 1e-10, btol = 1e-10,
	poles0 = 'linearized', f = None, g = None):
	r""" Vector fitting with a rank-one residue constraint


	Notes
	-----
	As of experiments on 31 Dec 2020, this iteration does not have the wonderful
	convergence properties of standard vector fitting.
	Sometimes this algorithm will converge, but in experiments with the ISS-1R model,
	the fixed points are almost always worse approximations than simply running
	vector fitting and then replacing the residues with their rank-one approximation. 

	Parameters
	----------
	X: array-like
		
	"""
	
	assert X.shape[1] == 1, "This only works with univariate data"
	assert len(Y.shape) == 3
	assert X.shape[0] == Y.shape[0], "Dimension of input and output must match"
	assert num_degree + 1 == denom_degree

	M, p, m = Y.shape

	if poles0 == 'linearized' or poles0 == 'vectorfit':
		
		# Use the linearized fit to construct an inital rational approximation
		if poles0 == 'linearized':
			rat = LinearizedRationalApproximation(num_degree, denom_degree)
		elif poles0 == 'vectorfit':
			rat = VectorFittingRationalApproximation(num_degree, denom_degree)

		rat.fit(X, Y)
		poles, residues = rat.pole_residue()

		# Replace the residues with their rank-1 approximations
		f = _zeros((denom_degree, p), X, Y, poles, residues)
		g = _zeros((denom_degree, m), X, Y, poles, residues)
		for k, R in enumerate(residues):
			U, s, VH = scipy.linalg.svd(R, full_matrices = False)
			f[k,:] = U[:,0]*np.sqrt(s[0])
			g[k,:] = VH[0,:].conj()*np.sqrt(s[0])
	else:
		# Pull in the manual initialization
		poles = np.array(poles0).flatten()
		assert len(poles) == denom_degree, "Number of poles does not match the desired degree"
		f = np.array(f)
		assert f.shape == (denom_degree, p), f"f does not match target shape ({denom_degree}, {p})"
		g = np.array(g)
		assert g.shape == (denom_degree, m), f"g does not match target shape ({denom_degree}, {m})"
	
	h = np.zeros(denom_degree)


	if verbose:
		printer = IterationPrinter(it = '4d', err = '20.14e', normh = '10.3e')
		printer.print_header(it = 'iter', err = '2-norm mismatch', normh = 'max |h|∞')
	
	if verbose:
		Yfit = eval_vecfit_rank(X, poles, f, g, h) 
		err0 = np.linalg.norm( (Y - Yfit).flatten())
		printer.print_iter(err = err0)	

	def update_poles(poles, h):
		return np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(len(poles)), h))
		
	
	for it in range(maxiter):
		
		# Solve for f (left vectors in numerator)
		C = _build_cauchy(X, poles)
		f, h = _fit_fh(Y, C, g)

		# Solve for g (right vectors in numerator)
		poles = update_poles(poles, h)
		C = _build_cauchy(X, poles)
		g, h = _fit_gh(Y, C, f)
		#f, g, h = _fit_fgh(Y, C, f, g)
		#print("f")
		#print(f)
		#print("g")
		#print(g)
		#print("h\n", h)

		# Rebalance f, g
		for k in range(denom_degree):
			fk_norm = np.linalg.norm(f[k])
			gk_norm = np.linalg.norm(g[k])
			
			f[k] *= np.sqrt(gk_norm/fk_norm)
			g[k] *= np.sqrt(fk_norm/gk_norm)
	

		Yfit = eval_vecfit_rank(X, poles, f, g, h) 
		err = np.linalg.norm( (Y - Yfit).flatten())
		poles = update_poles(poles, h)

		if verbose:
			printer.print_iter(it = it+1, err = err, normh = np.max(np.abs(h)) )

		#if np.max(np.abs(h)) < btol:
		#	break
