import numpy as np
from .sqp import *
from .arnoldi import *

def make_encoder(*args):
	def encode(*args_):
		return np.hstack([x.flatten() for x in args_])

	return encode

def make_decoder(*args):
	dims = [a.shape for a in args]

	def decode(x):
		out = []
		start = 0
		for dim in dims:
			stop = start + np.prod(dim)
			out += [ x[start:stop].reshape(*dim)]
			start = stop
		return out
			
	return decode
	
class RationalConstraint(NonlinearEqualityConstraint):
	r"""
	"""
	def __init__(self, X, Y, num_degree, denom_degree):
		self.num_degree = np.copy(num_degree)
		self.denom_degree = np.copy(denom_degree)
		self.num_basis = ArnoldiPolynomialBasis(X, self.num_degree)
		self.denom_basis = ArnoldiPolynomialBasis(X, self.denom_degree)
		
		self.P = self.num_basis.vandermonde_X
		a = np.zeros(list(self.P.shape[1:]))
		self.Q = self.denom_basis.vandermonde_X
		b = np.zeros(list(self.Q.shape[1:]))

		self.encode = make_encoder(a, b, Y)
		self.decode = make_decoder(a, b, Y)

	def fun(self, x):
		return self.fun_native(*self.decode(x))

	def fun_native(self, a, b, yfit):
		return self.P @ a - yfit * (self.Q @ b)
	  
	def jac(self, x):
		return self.jac_native(*self.decode(x))

	def jac_native(self, a, b, yfit):
		return np.hstack([self.P, -np.diag(yfit) @ self.Q, -np.diag(self.Q @ b)])
 
	def nullspace(self, x):
		return self.nullspace_native(*self.decode(x))	

	def nullspace_native(self, a, b, yfit):
		U1 = np.diag( 1./(self.Q @ b)) @ np.hstack([self.P, -np.diag(yfit) @ self.Q])
		U = np.vstack([np.eye(U1.shape[1]), U1])
		U, _ = np.linalg.qr(U, mode = 'reduced')
		return U

class RationalObjective:
	def __init__(self, x, y, a, b, num_degree, denom_degree):
		self.num_degree = np.copy(num_degree)
		self.denom_degree = np.copy(denom_degree)
		self.encode = make_encoder(a, b, y)
		self.decode = make_decoder(a, b, y)
		self.y = np.copy(y)
	
	def fun(self, x):
		return self.fun_native(*self.decode(x))

	def fun_native(self, a, b, yfit):
		return 0.5 * np.linalg.norm((yfit - self.y).flatten(), 2)**2

	def jac(self, x):
		return self.jac_native(*self.decode(x))

	def jac_native(self, a, b, yfit):
		return np.hstack([0*a, 0*b, (yfit - self.y).flatten()])

	def hess(self, x):
		return self.hess_native(*self.decode(x))

	def hess_native(self, a, b, yfit):
		return np.diag(
			np.hstack([0*a, 0*b, np.ones(self.y.flatten().shape)]))


class RatFitEquality(LiuYuanEqualitySQP):
	def __init__(self, x, y, num_degree, denom_degree, **kwargs):
		self._constraints = RationalConstraint(x, y, num_degree, denom_degree)
		a = np.zeros(self._constraints.P.shape[1])
		b = np.zeros(self._constraints.Q.shape[1])
	
		self._objective = RationalObjective(x, y, a, b, num_degree, denom_degree)

		LiuYuanEqualitySQP.__init__(self, self._objective.fun, self._objective.jac,
			self._objective.hess, self._constraints, **kwargs)

		self.x = self._objective.encode(a, b, y)
		self.z = np.zeros(y.shape).flatten()	

		self.encode = self._constraints.encode
		self.decode = self._constraints.decode

#	def solve_relaxation(self, x, h, A):
#		a, b, yfit = self.decode(x)
#		P = A[:,:len(a)]
#		Q = A[:,len(a):len(a)+len(b)]
		
	
	#def solve_qp(self, x, z, g, h, c, A, x0):
		 
			
