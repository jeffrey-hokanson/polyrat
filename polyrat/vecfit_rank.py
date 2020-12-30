r""" Rank-constrained vector fitting
"""

from itertools import product

import numpy as np
import scipy.linalg
from .arnoldi import ArnoldiPolynomialBasis
from .aaa import _build_cauchy


def _zeros(size, *args):
	r""" allocate a zeros matrix of the given size matching the type of the arguments
	"""
	if all([np.isrealobj(a) for a in args]):
		return np.zeros(size, dtype = np.float)
	else:
		return np.zeros(size, dtype = np.complex)


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
			Atmp[t*M:(t+1)*M,:] = C @ np.diag(g[:,t].conj()) #np.multiply(g[:t].conj(), C)

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
	M = Y.shape[0]
	r = C.shape[1]
	p = Y.shape[1]
	m = Y.shape[2]
	
	I = np.eye(m)

	# TODO: Make this a single loop function
	A = []
	for k, j in product(range(r), range(m)):
		Ajk = np.kron( C[:,k], np.outer(f[k], I[:,j]).flatten())
		A.append(Ajk)

	A.append(-Y.flatten()[:, None] * np.kron(C, np.ones((m*p,1))))
	
	A = np.column_stack(A)

	b = Y.flatten()
	x, res, rank, s = scipy.linalg.lstsq(A, b, overwrite_a = True, overwrite_b = False)	

	g = x[:r*m].reshape(r, m).conj()
	h = x[r*m:]
	return g, h
	


def eval_vecfit_rank(X, poles, f, g, h):
	Y = np.zeros((X.shape[0], f.shape[1], g.shape[1]), dtype = np.complex)


	denom = np.ones(X.shape[0], dtype = np.complex)
	for k in range(len(poles)):
		Y += np.einsum('i,jk->ijk', 1/(X[:,0] - poles[k]), np.outer(f[k], g[k].conj()))
		denom += h[k]/(X[:,0] - poles[k])

	Y *= 1./denom[:,None, None]

	return Y	

def vecfit_rank(X, Y, num_degree, denom_degree, 
	verbose = True, maxiter = 500, Basis = ArnoldiPolynomialBasis,
	poles0 = 'linearized', ftol = 1e-10, btol = 1e-7):
	r""" Vector fitting with a rank constraint
	"""
	
	assert X.shape[1] == 1, "This only works with one-dimensional data"
	assert len(Y.shape) == 3
	assert X.shape[0] == Y.shape[0], "Dimension of input and output must match"
	assert num_degree + 1 == denom_degree

	# TODO: initial poles	
	poles = poles0

	f = np.random.randn(denom_degree,Y.shape[1])
	g = np.random.randn(denom_degree,Y.shape[2])
	h = np.zeros(len(poles))
	#f = np.ones((denom_degree, Y.shape[1]))
	#g = np.ones((denom_degree, Y.shape[2])) 
	#g = 1 + np.random.randn(denom_degree,Y.shape[2])**2
	
	Yfit = eval_vecfit_rank(X, poles, f, g, h) 
	err1 = np.linalg.norm( (Y - Yfit).flatten())
	print("initial", err1)

	C = _build_cauchy(X, poles)
	for it in range(1):
		f = _fit_f(Y, C, g)
		Yfit = eval_vecfit_rank(X, poles, f, g, h) 
		err1 = np.linalg.norm( (Y - Yfit).flatten())
		print("f update", err1)

		g = _fit_g(Y, C, f)		
		Yfit = eval_vecfit_rank(X, poles, f, g, h) 
		err1 = np.linalg.norm( (Y - Yfit).flatten())
		print("g update", err1)
#		for k in range(len(poles)):
#			print(np.outer(f[k], g[k]))


	for it in range(maxiter):

		# Solve for f (left vectors in numerator)
		C = _build_cauchy(X, poles)
		#f, h = _fit_fh(Y, C, g)
		#poles = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(len(poles)), h))
		#C = _build_cauchy(X, poles)
		f = _fit_f(Y, C, g)
		
		# Solve for g (right vectors in numerator)
		g, h = _fit_gh(Y, C, f)
		Yfit = eval_vecfit_rank(X, poles, f, g, h) 
		err = np.linalg.norm( (Y - Yfit).flatten())
		poles = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(len(poles)), h))

		print(f"\n------ iter {it} -------")
		print("error", err)
		print("h", np.abs(h))

		print("poles")
		for p in poles[np.argsort(poles.imag)]:
			print(p)

		if np.max(np.abs(h)) < btol:
			break
