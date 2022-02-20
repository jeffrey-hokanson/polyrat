import numpy as np
import abc
from iterprinter import IterationPrinter


class Termination(Exception):
	pass
class OptimalTermination(Termination):
	def __repr__(self):
		return 'Optimal point'
class FritzJohnPoint(OptimalTermination):
	def __repr__(self):
		return 'Fritz John point'

class InfeasibleTermination(Termination):
	def __repr__(self):
		return 'Infeasible point'

class SmallStepTermination(Termination):
	def __init__(self, norm_dx):
		self.norm_dx = float(norm_dx)
	def __repr__(self):
		return f"SmallStepTermiation({self.norm_dx:5e})"


class Constraint(abc.ABC):
	
	def __call__(self, x):
		return self.fun(x)
	
	@abc.abstractmethod
	def fun(self, x):
		raise NotImplementedError

	@abc.abstractmethod
	def jac(self, x):
		raise NotImplementedError
	
	@abc.abstractmethod
	def hess(self, x):
		r"""
		"""
		raise NotImplementedError
	
	@abc.abstractmethod
	def hess_vec(self, x, z):
		r"""
		"""
		raise NotImplementedError


class NonlinearEqualityConstraint(Constraint):
	def __init__(self, fun, jac = None, hess = None, hess_vec = None, target = None):
		if target is None:
			self._fun = fun
		else:
			self._fun = lambda x: fun(x) - target

		self._jac = jac
		self._hess = hess
		self._hess_vec = hess_vec
	
	def fun(self, x):
		return self._fun(x)

	def jac(self, x):
		if self._jac is not None:
			return self._jac(x)
		raise NotImplementedError

	def hess(self, x):
		if self._hess is not None:
			return self._hess(x)
		raise NotImplementedError

	def hess_vec(self, x, z):
		if self._hess_vec is not None:
			return self._hess_vec(x, z)
		raise NotImplementedError

	def orthogonal_nullspace(self, A):
		r""" Compute the nullspace of the constraint linearization

		Parameters
		----------
		A: np.ndarray
			Existing value of linearized constraints; i.e., A = self.jac(x)
		"""
		# This is copied from the SciPy implementation (but using numpy to avoid dependency) 
		u, s, vh = np.linalg.svd(A, full_matrices=True)
		M, N = u.shape[0], vh.shape[1]
		rcond = np.finfo(s.dtype).eps * max(M, N)
		tol = np.amax(s) * rcond
		num = np.sum(s > tol, dtype=int)
		Q = vh[num:,:].T.conj()
		return Q


class Objective(abc.ABC):
	def __init__(self, fun, jac = None, hess = None):
		self._fun = fun
		self._jac = jac
		self._hess = hess	

	def __call__(self, x):
		return self.fun(x)
	
	def fun(self, x):
		return self._fun(x)

	def jac(self, x):
		if self._jac is not None:
			return self._jac(x)

		raise NotImplementedError
	
	def hess(self, x):
		if self._hess is not None:
			return self._hess(x) 

		raise NotImplementedError


class EqualitySQP(abc.ABC):
	def __init__(self, objective, constraint,
			tol_dx = 1e-6, 
			tol_h = 1e-6,
			tol_Ah = 1e-6,
			tol_opt = 1e-6
		):
		self.objective = objective
		self.constraint = constraint
		self.tol_dx = tol_dx
		self.tol_h = tol_h
		self.tol_Ah = tol_Ah
		self.tol_opt = tol_opt
		self.callbacks = []

	def translation(self, x, p, alpha):
		r""" Translate the initial point x in the direction p by length alpha
	
		In most situations, with most solvers 
	
		Parameters
		----------
		x: np.ndarray
			Starting point
		p: np.ndarray
			Direction 
		alpha: float
			Length to travel from x along p
		"""
		return x + alpha * p

	def verbose_callback(self, state):
		raise NotImplementedError

	def run_callbacks(self, state):
		for callback in self.callbacks:
			callback(state)

	@abc.abstractmethod
	def step(self, x, z, **kwargs):
		r""" Perform one step of optimization

		Parameters
		----------
		x: np.ndarray
			Current solution estimate
		z: np.ndarray
			Current Lagrange multiplier
		""" 
		pass


	def init_solver(self):
		pass

	def solve(self, x0, z0 = None, maxiter = 100, verbose = True):
		self.init_solver()

		if verbose and not any([callback == self.verbose_callback for callback in self.callbacks]):
			self.callbacks += [self.verbose_callback]

		if z0 is None:
			z = 0*self.constraint.fun(x0)
		x = x0

		for it in range(maxiter):
			try:
				x, z = self.step(x, z, it = it)
			except Termination as e:
				self.x = x
				self.z = z
				if verbose:
					print(repr(e))
				break
		return self.x


	def check_termination(self, norm_lagrangian_grad, norm_h, norm_Ah):
		if ((norm_lagrangian_grad < self.tol_opt) and (norm_h < self.tol_h)):
			raise OptimalTermination
		if (norm_h < self.tol_h) and (norm_Ah < self.tol_Ah * min(norm_h, 1)):
			raise FritzJohnPoint
#		if (norm_h > self.tol_h) and (norm_Ah < self.tol_Ah * ):
	
	def solve_relaxation(self, h, A):
		r""" Compute direction minimizing constraint violation

		Approximately solve 

			min_p \| h + A @ p \|_2

		where the constraint h(x + p) \approx h + A @ p

		Parameters
		----------
		h: np.ndarray
			Value of constraints at current iterate
		A: np.ndarray
			Gradient of constraints at current iterate
		
		Returns
		-------
		p: np.ndarray
			Solution approximately minimizing the least squares problem 
		"""
		return np.linalg.lstsq(A, -h, rcond = None)[0]


	def solve_qp(self, g, c, A, B, x0 = None, z0 = None):
		r""" Solve the quadratic program
		
		Approximately solve the quadratic subproblem

		min_p    g.T @ p + 0.5 * (p.T @ B @ p)
		s.t.     A @  = c

		This corresponds to solving the KKT system

			[B     A.T ] [p] = [-g]
			[A     0   ] [z] = [c]

		Parameters
		----------
		g: np.ndarray
			gradient vector

		Returns
		-------
		p: np.ndarray
			Search direction
		z: np.ndarray
			New Lagrange multipliers

		"""
		# Number of constraints
		n_con = A.shape[0]

		# Build KKT system
		Z = np.zeros((n_con, n_con))
		AA = np.block([[B, A.T],[A, Z]])
		bb = np.hstack([-g, c])

		# Solve KKT system
		xx = np.linalg.solve(AA, bb)

		# Partition into state and Lagrange multipliers
		return xx[:-n_con], xx[-n_con:] 
			 


class ReducedHessian(EqualitySQP):
	r""" Solve the quadratic subproblem using the reduced Hessian approach 
	
	See: Nocedal and Wright
	"""
	def solve_qp(self, g, c, A, B, x0 = None, z0 = None):
		# Basis for the nullspace of the linearized constraints
		U = self.constraint.orthogonal_nullspace(A)

		# TODO: Migrate to Scipy and compute one QR factorization of A? 
		p_range = np.linalg.lstsq(A, c, rcond = None)[0]

		# Reduced Hessian
		H = U.T @ B @ U
		p_null = U @ np.linalg.solve(U.T @ B @ U, -U.T @ B @ p_range - U.T @ g)
		p = p_range + p_null

		z = np.linalg.lstsq(A.T, -g - B @ p, rcond = None)[0]
		return p, z

class LiuYuanEqualitySQPVerbose(IterationPrinter):
	def __init__(s):
		super().__init__(
			it = '4d', 
			obj = '20.10e', 
			con = '8.2e', 
			lagrange = '8.2e',	
			norm_dx = '8.2e',
			alpha = '8.2e',
		)	
	def __call__(s, state):
		if 'it' in state and state['it'] == 0:
			s.print_header(
				it = 'iter', 
				obj = 'objective',
				con = 'constraint',
				lagrange = 'optimality',
				norm_dx = '|| dx ||',
				alpha = 'step',
			)
		s.print_iter(
			it = state.get('it', None),
			obj = state.get('f1', state.get('f0', None)),
			con = state.get('v1', state.get('v0', None)),
			lagrange = state.get('norm_lagrangian_grad', None),
			norm_dx = state.get('norm_dx', None),
			alpha = state.get('alpha', None),
		)

class LiuYuanEqualitySQP(EqualitySQP):
	r"""
	"""
	def __init__(self, objective, constraint,
		sigma = 0.01, xi1 = 1e-10, xi2 = 1e-4, kappa1 = 1e5,
		kappa2 = 1e-6,**kwargs,):

		super().__init__(objective, constraint, **kwargs)
		self.sigma = sigma
		self.xi1 = xi1	
		self.xi2 = xi2	
		self.kappa1 = kappa1
		self.kappa2 = kappa2

		self.verbose_callback = LiuYuanEqualitySQPVerbose()

	
	def init_solver(self):
		self.r = 0.9
		self.vmax = 0
		self.eq3_old = False

	def step(self, x, z, **kwargs):
		f0 = self.objective.fun(x)
		h = self.constraint.fun(x)
		v0 = np.linalg.norm(h)
		A = self.constraint.jac(x)
		g = self.objective.jac(x)
		
		#############################################################################
		# Check termination conditions
		#############################################################################
		lagrangian_grad = g + A.T @ z
		norm_lagrangian_grad = np.linalg.norm(lagrangian_grad)
		norm_h = v0
		norm_Ah = np.linalg.norm(A.T @ h)

		try:
			self.check_termination(norm_lagrangian_grad, norm_h, norm_Ah)
		except Termination as e:
			self.run_callbacks({**locals(), **kwargs} )
			raise e
		#############################################################################
		# Compute the relaxation step 
		#############################################################################
		dp = self.solve_relaxation(h, A)
		
		#############################################################################
		# Solve the QP subproblem 
		#############################################################################
		
		B = self.objective.hess(x)
		# Add contribution constraint Hessian if avalible
		try: B += self.constraint.hess_vec(x, z)
		except NotImplementedError: pass
	
		# Solve relaxed QP subproblem 
		p, z_new = self.solve_qp(g, A @ dp, A, B, x0 = dp, z0 = z)

		# Lagrange multplier search direction
		p_z = z_new - z

		#############################################################################
		# Backtracking line search
		#############################################################################

		# In LY11, 
		# v(x) = \| h(x)\|_2
		# phi(x;d) = \| h(x) + A(x) @ d\| - \|h(x)\|
		norm_p = np.linalg.norm(p)
		v0 = norm_h
		phi0 = np.linalg.norm( h + A @ p) - norm_h
		gp = g.T @ p 
	
		alpha = 1
		while True:
			x1 = self.translation(x, p, alpha)
			f1 = self.objective.fun(x1)
			v1 = np.linalg.norm(self.constraint.fun(x1))
			
			eq1 = (f1 - f0) <= min(self.sigma * alpha * gp, -self.xi1 * v1)
			eq2 = (v1 <= max( (self.r +1)/2, 0.95) * self.vmax) or (self.vmax == 0)
			eq3 = (v1 - v0) <= min(self.sigma * alpha * phi0, -self.xi2 * alpha**2 * norm_p**2)
			
			norm_dx = np.linalg.norm(x1 - x)

			if ((eq1 and eq2) or eq3):
				break

			if norm_dx < self.tol_dx:
				self.run_callbacks({**locals(), **kwargs} )
				raise SmallStepTermination(norm_dx)
			
			alpha *= 0.5	

		# Update the optimizer state constants 

		if eq3:
			self.r = v1/v0
		if eq3 and not self.eq3_old:
			self.vmax = v1

		self.eq3_old = eq3	

		z1 = z + alpha * p_z

		self.run_callbacks({**locals(), **kwargs} )

		return x1, z1

