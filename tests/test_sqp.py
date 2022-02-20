import numpy as np
from polyrat.sqp import *

def test_sqp():
	constraint = NonlinearEqualityConstraint(
		fun = lambda x: np.array([x[0]**2 + x[1]**2 - 1,]),
		jac = lambda x: 2*x.reshape(1,2),
		hess_vec = lambda x, z: z*np.eye(2),
	)
	x = np.ones(2)/np.sqrt(2)
	print(constraint.fun(x))
	print(constraint.jac(x))
	print(constraint.hess_vec(x, 1))
	objective = Objective(
		fun = lambda x:  2*(x[0]**2 + x[1]**2 -1) - x[0],
		jac = lambda x: np.array([4*x[0] - 1, 4*x[1]]),
		hess = lambda x: 4*np.eye(2)
	)

	solver = LiuYuanEqualitySQP(objective, constraint)
	z0 = np.ones(1)
	x0 = np.ones(2)/np.sqrt(2)
	print(x0, constraint.fun(x0))
	solver.init_solver()
	print(solver.step(x0, z0))	
	print(solver.solve(x0))	

	class ReducedSQP(ReducedHessian, LiuYuanEqualitySQP):
		pass

	solver_reduced = ReducedSQP(objective, constraint)
	solver_reduced.solve(x0)

	# Check the reduced Hessian

	for it in range(10):	
		x1, z1 = solver.step(x0, z0)
		x2, z2 = solver_reduced.step(x0, z0)
		print(x1, x2)
		print(z1, z2)
		x0, z0 = x1, z1
		assert np.allclose(x1, x2)
		assert np.allclose(z1, z2)

if __name__ == '__main__':
	test_sqp()	
