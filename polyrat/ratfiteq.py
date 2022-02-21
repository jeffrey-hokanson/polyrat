import numpy as np
from .sqp import *
from .arnoldi import *
import scipy.linalg

class Coding:
	def __init__(self, *args):
		self.dims = [a.shape for a in args]

	def encode(self, *args):
		return np.hstack([x.flatten() for x in args])
	
	def decode(self, x):
		out = []
		start = 0
		for dim in self.dims:
			stop = start + np.prod(dim)
			out += [ x[start:stop].reshape(*dim)]
			start = stop
		return out

class ConstantRationalConstraint(NonlinearEqualityConstraint):
	r"""
	"""
	def __init__(self, x, y, num_degree, denom_degree):
		self.num_degree = np.copy(num_degree)
		self.denom_degree = np.copy(denom_degree)

		self.num_basis = ArnoldiPolynomialBasis(x, self.num_degree)
		self.denom_basis = ArnoldiPolynomialBasis(x, self.denom_degree)
		
		self.P = self.num_basis.vandermonde_X
		a = np.zeros(list(self.P.shape[1:]))
		self.Q = self.denom_basis.vandermonde_X
		b = np.zeros(list(self.Q.shape[1:]))
		
		self.coding = Coding(a, b, y)

	def fun(self, x):
		return self.fun_native(*self.coding.decode(x))

	def fun_native(self, a, b, y):
		return self.P @ a - y * (self.Q @ b)
	  
	def jac(self, x):
		return self.jac_native(*self.coding.decode(x))

	def jac_native(self, a, b, yfit):
		return np.hstack([self.P, -np.diag(yfit) @ self.Q, -np.diag(self.Q @ b)])
 
	def orthogonal_nullspace(self, A):
		n_a = np.prod(self.coding.dims[0])
		n_b = np.prod(self.coding.dims[1])
		n_y = np.prod(self.coding.dims[2])

		P = A[:,:n_a]
		yQ = A[:,n_a:n_a+n_b]
		diagQb = A[:, n_a+n_b:]
		
		Qb = np.diag(diagQb)
		U1 = -(1./Qb)[:, None] * A[:,:n_a+n_b]
		U1 = np.vstack([np.eye(U1.shape[1]), U1])
		U, _ = np.linalg.qr(U1, mode = 'reduced')
		return U


class RationalObjective:
	def __init__(self, ydata, coding):
		self.coding = coding
		self.ydata = np.copy(ydata)
	
	def fun(self, x):
		return self.fun_native(*self.coding.decode(x))

	def fun_native(self, a, b, y):
		return 0.5 * np.linalg.norm((y - self.ydata).flatten(), 2)**2

	def jac(self, x):
		return self.jac_native(*self.coding.decode(x))

	def jac_native(self, a, b, y):
		return np.hstack([0*a, 0*b, (y - self.ydata).flatten()])

	def hess(self, x):
		return self.hess_native(*self.coding.decode(x))

	def hess_native(self, a, b, yfit):
		return np.diag(
			np.hstack([0*a, 0*b, np.ones(self.ydata.flatten().shape)]))



class RatFitEquality(LiuYuanEqualitySQP):
	def __init__(self, x, y, num_degree, denom_degree, **kwargs):
		constraint = ConstantRationalConstraint(x, y, num_degree, denom_degree)
		objective = RationalObjective(y, constraint.coding)

		super().__init__(objective, constraint, **kwargs)

	@property
	def ydata(self):
		return self.objective.ydata

	@property
	def coding(self):
		return self.constraint.coding

	def initialization(self):
		P, Q = self.constraint.P, self.constraint.Q
		Q_P, R_P = np.linalg.qr(P)
		A = np.multiply(self.ydata[:, None], Q)
		A -= Q_P @ (Q_P.conj().T @ A)

		U, s, VH = np.linalg.svd(A, full_matrices = False)	
		b0 = VH.T.conj()[:,-1]
		rhs = Q_P.conj().T @ (self.ydata * (Q @ b0))
		a0 = scipy.linalg.solve_triangular(R_P, rhs)
		norm_b = np.linalg.norm(b0)
		a0 /= norm_b
		b0 /= norm_b
		y = (P @ a0)/(Q @ b0)
		return self.coding.encode(a0, b0, y)

	def translation(self, x, p, alpha):
		a, b, y = self.coding.decode(x)
		p_a, p_b, p_y = self.coding.decode(p)

		# Use a Grassman step for b
		#norm_b = np.linalg.norm(b)
		#print(norm_b)
		#a /= norm_b
		#b /= norm_b

		# Tangent to space
		#p_b -= b * (b.conj().T @ p_b)

		# transport 
		#norm_p_b = np.linalg.norm(p_b)
		#b_new = b * np.cos(norm_p_b * alpha) + p_b/norm_p_b * np.sin(norm_p_b * alpha)
		a_new = a + p_a * alpha
		b_new = b + p_b * alpha
		y_new = y + p_y * alpha

		b_norm = np.linalg.norm(b_new)
		a_new /= b_norm
		b_new /= b_norm
	
		#print("a", a_new)
		#print("b", b_new)
		#print("y", y_new)
		return self.coding.encode(a_new, b_new, y_new)
	
	def solve(self, **kwargs):
		x0 = self.initialization()
		super().solve(x0, **kwargs)			
		
			
