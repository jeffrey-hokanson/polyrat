r""" Utilties for Sanathanan-Koerner style iterations
"""



import numpy as np
from scipy.sparse.linalg import LinearOperator


def linearized_ratfit_operator_dense(P, Q, Y):
	r""" Dense analog of LinearizedRatfitOperator
	"""
	nout = int(np.prod(Y.shape[1:]))

	A = np.kron(np.eye(nout), P)
	if nout == 1:
		At = -np.multiply(Y.reshape(-1,1), Q)
	else:
		At = np.vstack([
			-np.multiply(Y[(slice(P.shape[0]),*idx)].reshape(-1,1), Q)
			for idx in np.ndindex(Y.shape[1:])
			])
		
	A = np.hstack([A, At])
	return A

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
		print("Y", self.Y.shape)
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




