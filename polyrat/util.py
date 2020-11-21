r""" Utilties for Sanathanan-Koerner style iterations
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
import scipy.linalg


def linearized_ratfit_operator_dense(P, Q, Y):
	r""" Dense analog of LinearizedRatfitOperator
	"""
	nout = int(np.prod(Y.shape[1:]))

	M = P.shape[0]
	A = np.kron(np.eye(nout), P)
	if nout == 1:
		At = -np.multiply(Y.reshape(-1,1), Q)
	else:
		At = np.vstack([
			-np.multiply(Y[(slice(M),*idx)].reshape(-1,1), Q)
			for idx in np.ndindex(Y.shape[1:])
			])
		
	A = np.hstack([A, At])
	return A


def minimize_2norm_dense(P, Q, Y):
	A = linearized_ratfit_operator_dense(P, Q, Y)
	U, s, VH = scipy.linalg.svd(A, full_matrices = False, overwrite_a = True)
		
	# Condition number of singular vectors, cf. Stewart 01: Eq. 3.16
	with np.errstate(divide = 'ignore'):
		cond = s[0]*np.sqrt(2)/(s[-2] - s[-1])
	
	x = VH.T.conj()[:,-1]
	b = x[-Q.shape[1]:]
	m = P.shape[1]
	a = np.zeros((m, *Y.shape[1:]), dtype = x.dtype)
	for j, idx in enumerate(np.ndindex(Y.shape[1:])):
		a[(slice(m),*idx)] = x[j*m:(j+1)*m]
	return a, b, cond

def minimize_2norm_varpro(P, Q, Y, P_orth = False, method = 'svd'):
	r"""

	Parameters
	----------
	method: ['svd', 'ls']
		How to compute the denominator.
		If 'svd', it uses the singular value decomposition;
		if 'ls', it pins b[0]=0 and solves the least squares problem.
			
	"""
	M = Y.shape[0]
	m = P.shape[1]
	n = Q.shape[1]
	nout = int(np.prod(Y.shape[1:]))
	if P_orth:
		Q_P = P
	else:
		Q_P, R_P = np.linalg.qr(P, mode = 'reduced')
	

	# Form the matrix 
	# [P P^* - I] diag(y) Q
	if nout == 1:
		A = np.multiply(Y.reshape(-1,1), Q)
		A -= Q_P @ (Q_P.conj().T @ A)
	else:
		A = []
		for idx in np.ndindex(Y.shape[1:]):
			At = np.multiply(Y[(slice(M),*idx)].reshape(-1,1), Q)
			A.append(At - Q_P @ (Q_P.conj().T @ At))
		A = np.vstack(A)	

	if method == 'svd':
		U, s, VH = np.linalg.svd(A, full_matrices = False)

		# Condition number of singular vectors, cf. Stewart 01: Eq. 3.16
		with np.errstate(divide = 'ignore'):
			cond = s[0]*np.sqrt(2)/(s[-2] - s[-1])
	
		b = VH.T.conj()[:,-1]
	else:
		x, _, _, s = np.linalg.lstsq(A[:,1:], -A[:,0], rcond = None)
		b = np.hstack([[1], x])
		cond = s[0]/s[-1]

	a = np.zeros((m, *Y.shape[1:]), dtype = b.dtype)
	Qb = Q @ b
	for j, idx in enumerate(np.ndindex(Y.shape[1:])):
		x = Q_P.conj().T @ (Y[(slice(M),*idx)] * Qb)
		if P_orth:
			a[(slice(m),*idx)] = x
		else:	
			a[(slice(m),*idx)] = scipy.linalg.solve_triangular(R_P, x)

	return a, b, cond


class LinearizedRatfitOperator(LinearOperator):
	r"""A linear operator in many Sanathanan-Koerner style algorithms for array-valued problems.

	Many algorithms that solve rational approximation by "linearizing"
	by multiplying through the denominator.
	In the scalar-valued case this yields an
	minimization problem 
	
	.. math::
		
		\min_{\mathbf{a}, \mathbf{b} \ne \mathbf{0} } \left\|
			\begin{bmatrix}
				\mathbf{P} &  -\textrm{diag}(\mathbf{y}) \mathbf{Q}
			\end{bmatrix}
			\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}
		\right\|_2.

	As :math:`\mathbf{P}` and :math:`\mathbf{Q}` are dense,
	we simply solve this problem using a dense SVD. 

	In the array-valued setting, we need to solve a larger, sparse system

	.. math::
		\min_{\mathbf{a}^{(1)}, \mathbf{a}^{(2)}, \ldots, \mathbf{a}^{(N)}, \mathbf{b}}
		\left\|
		\begin{bmatrix} 
			\mathbf{P} & & & & 
				-\textrm{diag}(\mathbf{y}^{(1)}) \mathbf{Q} \\
			& \mathbf{P} & & &
				-\textrm{diag}(\mathbf{y}^{(2)}) \mathbf{Q} \\
			& &  \ddots & & \vdots \\
			&  & & \mathbf{P} & 
				-\textrm{diag}(\mathbf{y}^{(N)}) \mathbf{Q} 
		\end{bmatrix}
		\begin{bmatrix}
			\mathbf{a}^{(1)} \\ \mathbf{a}^{(2)} \\ \vdots \\ \mathbf{a}^{(N)} \\ \mathbf{b}
		\end{bmatrix}
		\right\|_2.


	This class implements a :class:`~scipy:scipy.sparse.linalg.LinearOperator` representing this
	block sparse matrix for use with iterative SVD algorithms.


	Parameters
	----------
	P: :class:`~numpy:numpy.ndarray`
		Basis for numerator polynomial
	Q: :class:`~numpy:numpy.ndarray`
		Basis for denominator polynomial
	Y: :class:`~numpy:numpy.ndarray`
		Data trying to be fit.
	"""
	def __init__(self, P, Q, Y):
		self.P = np.array(P)
		self.Q = np.array(Q)
		self.Y = np.array(Y)
		assert self.Y.shape[0] == self.P.shape[0] == self.Q.shape[0], "Wrong dimensions"

	@property
	def shape(self):
		nout = int(np.prod(self.Y.shape[1:]))
		return (self.P.shape[0]*nout, self.P.shape[1]*nout + self.Q.shape[1]) 


	@property
	def dtype(self):
		return (self.P[0,0] * self.Q[0,0] * self.Y.flat[0]).dtype

	def _matmat(self, X):
		if any([np.iscomplexobj(self.P), 
				np.iscomplexobj(self.Q), 
				np.iscomplexobj(self.Y),
				np.iscomplexobj(X)]):
			Z = np.zeros((self.shape[0], X.shape[1]), dtype = np.complex)
		else:
			Z = np.zeros((self.shape[0], X.shape[1]), dtype = np.float)
		
		M = self.Y.shape[0]
		m = self.P.shape[1]
		n = self.Q.shape[1]

		iterator = np.ndindex(self.Y.shape[1:])

		nout = int(np.prod(self.Y.shape[1:]))
		for j, idx in enumerate(iterator):
			Z[j*M:(j+1)*M,:] = self.P @ X[j*m:(j+1)*m] \
				- np.multiply(self.Y[(slice(M),*idx)].reshape(-1,1), self.Q @ X[-n:])

		return Z

	def _rmatmat(self, X):
		if any([np.iscomplexobj(self.P), 
				np.iscomplexobj(self.Q), 
				np.iscomplexobj(self.Y),
				np.iscomplexobj(X)]):
			Z = np.zeros((self.shape[1], X.shape[1]), dtype = np.complex)
		else:
			Z = np.zeros((self.shape[1], X.shape[1]), dtype = np.float)


		M = self.Y.shape[0]
		m = self.P.shape[1]
		n = self.Q.shape[1]

		iterator = np.ndindex(self.Y.shape[1:])

		nout = int(np.prod(self.Y.shape[1:]))
		PH = self.P.conj().T
		QH = self.Q.conj().T
		for j, idx in enumerate(iterator):
			Z[j*m:(j+1)*m,:] = PH @ X[j*M:(j+1)*M] 
			Z[-n:,:] -= QH @ np.multiply(self.Y[(slice(M),*idx)].conj()[:,np.newaxis], X[j*M:(j+1)*M])

		return Z
	
	def _rmatvec(self, x):
		return self._rmatmat(x.reshape(-1,1)).flatten()



def minimize_2norm_sparse(P, Q, Y):
	A = LinearizedRatfitOperator(P, Q, Y)

	# For unclear reasons, seeking two singular vectors yields inaccurate 
	# right eigenvectors
#	if compute_cond:
#		U, s, VH = svds(A, k = 2, tol = 0, which = 'SM')
#		sm = np.min(s)
#		sm1 = np.max(s)
#		print(VH.T.conj())
#		x = VH.T.conj()[:,int(np.argmin(s))]
#		print(x)
#		
#		_, s0, _ = svds(A, k = 1, which = 'LM')
#		# Condition number of singular vectors, cf. Stewart 01: Eq. 3.16
#		with np.errstate(divide = 'ignore'):
#			cond = s0*np.sqrt(2)/(sm1 - sm)

	cond = None
	U, s, VH = svds(A, k=1, tol = 0, which = 'SM')
	x = VH.T.conj()[:,0]
	Ax = A @ x
	print(np.linalg.norm(Ax, 2), s)
	assert np.isclose(np.linalg.norm(Ax,2), s, atol = 1e-7), "Incorrect estimate of smallest singular value"

	b = x[-Q.shape[1]:]
	m = P.shape[1]
	a = np.zeros((m, *Y.shape[1:]), dtype = x.dtype)
	for j, idx in enumerate(np.ndindex(Y.shape[1:])):
		a[(slice(m),*idx)] = x[j*m:(j+1)*m]
	return a, b, cond


