""" Sequential linear programing approach for minimax optimization

"""

import numpy as np
from iterprinter import IterationPrinter
from scipy.optimize import OptimizeResult, approx_fprime, linprog, Bounds
from scipy.optimize._constraints import new_bounds_to_old
import cvxopt as co

def slp(fun, x0, args = (), jac = None, bounds = None, constraints = (), hess = None, hessp = None, verbose = True,
	maxiter = 100, xtol = 1e-6, c_armijo = 0.01, bt_maxiter = 30, **options):
	r""" Sequential linear programming approach for constrained optimization problems

	Solves 
	
	min_x  f(x)
	s.t.   lbc <= c(x) <= ubc 
           lb <= x <= ub

	"""

	nfev = 0
	njev = 0

	if jac is None or jac is False:
		assert False

	# Normalize constraints format as a list
	if not isinstance(constraints, list):
		constraints = [constraints]

	if verbose:
		iterprinter = IterationPrinter(it = '4d', obj = '20.14e', alpha = '8.3e', delta = '8.3e')
		iterprinter.print_header(it = 'iter', obj = "objective fun.", alpha = 'step length', delta = 'trust region radius')

	x = np.copy(x0)
	delta = 1e0

	stop = False
	it = 0

	fx = fun(x)


	if verbose:
		iterprinter.print_iter(it = it, obj = fx)

	while it < maxiter and not stop:
		it += 1
		# Generate linearized constraints
		Acon = []
		bcon = []
		for con in constraints:
			fx = con.fun(x)
			Jx = con.jac(x)

			# fx + Jx @ p <= ub
			I = np.isfinite(con.ub)
			Acon.append(Jx[I])
			bcon.append(con.ub[I] - fx[I])
			# lb <= fx + Jx @ p
			I = np.isfinite(con.lb)
			Acon.append(-Jx[I])
			bcon.append(fx[I] - con.lb[I])	


		fx = fun(x)
		# Generate search direction
		g = jac(x)	
		
		# Convert the bounds into a format for linprog
		if isinstance(bounds, Bounds):
			#linprog_bounds = [(lbi, ubi) for lbi, ubi in zip(bounds.lb, bounds.ub)]
			linprog_bounds = new_bounds_to_old(np.maximum(bounds.lb - x, -delta), np.minimum(bounds.ub - x, delta), len(x0)) 
		elif bounds is None:
			# Since Linprog assumes non-negative, we have to specify infinite lower bounds 
			linprog_bounds = [(None, None) for i in enumerate(x0)]	
		else:
			raise NotImplementedError
			#linprog_bounds = bounds

		# Solve LP
		if False:
			res = linprog(g, A_ub = np.vstack(Acon), b_ub = np.hstack(bcon), bounds = linprog_bounds)
			if not res.success:
				if verbose: print("linear program failed with message", res.message)
				return OptimizeResult(
						x = x, 
						success = False,
						status = -1,
						message = 'LP solve failed with message ' + res.message,
						fun = fx,
						jac = g, 	
						nit = it,
						nfev = nfev,
						njev = njev,)

			p = res.x
		else:
			co.solvers.options['show_progress'] = False
			G = co.matrix(np.vstack(Acon + [np.eye(len(x)), -np.eye(len(x))] ))
			h = co.matrix(np.hstack(bcon + [b[1] for b in linprog_bounds] + [-b[0] for b in linprog_bounds])) 
			sol = co.solvers.lp(co.matrix(g),G, h)
			#print(sol)
			p = np.array(sol['x']).flatten()

	
		# Backtracking line search	
		pred_decrease = g.T @ p
		for alpha in [0.5**k for k in range(bt_maxiter)]:
			xn = x + alpha*p
			fxn = fun(xn)
			# Check if satisfy Armijo conditions
			improved = (fxn	<= fx + c_armijo*alpha*pred_decrease)	
			feasible = True
			for con in constraints:
				cxn = con.fun(xn)
				feasible = feasible & (np.all(cxn <= con.ub) and np.all(con.lb <= cxn))
			
			if feasible and improved:
				break

		if not np.isclose(alpha,1):
			delta = np.max(np.abs(alpha*p))
		else:
			delta *=2
		

		if not (feasible and improved):
			return OptimizeResult(
					x = x, 
					success = False,
					status = -2,
					message = 'unable to find a successful step',
					fun = fx,
					jac = g, 	
					nit = it,
					nfev = nfev,
					njev = njev,)
				
		if verbose:
			iterprinter.print_iter(it=it, obj = fxn, alpha =alpha, delta = delta)

		if np.max(np.abs(alpha*p)) < xtol:
			if verbose: print("small change in parameters")
			return OptimizeResult(
					x = x, 
					success = True,
					status = 0,
					message = 'small change in objective',
					fun = fx,
					jac = g, 	
					nit = it,
					nfev = nfev,
					njev = njev,)
		x = xn
			
	return OptimizeResult(
			x = x, 
			success = True,
			status = 0,
			message = 'maximum number of iterations exceeded',
			fun = fx,
			jac = g, 	
			nit = it,
			nfev = nfev,
			njev = njev,)


def sl1lp(fun, x0, args = (), jac = None, bounds = None, constraints = (), verbose = True,
	maxiter = 100, **options):
	r"""

	Solve a constrained optimization problem in the l1 relaxation (see, Nocedal and Wright Sec. 18.5)

	""" 

	if jac is None or jac is False:
		assert False

	# Convert the bounds into a format for linprog
	if isinstance(bounds, Bounds):
		#linprog_bounds = [(lbi, ubi) for lbi, ubi in zip(bounds.lb, bounds.ub)]
		linprog_bounds = new_bounds_to_old(bounds.lb, bounds.ub, len(x0)) 
	elif bounds is None:
		# Since Linprog assumes non-negative, we have to specify infinite lower bounds 
		linprog_bounds = [(None, None) for i in enumerate(x0)]	
	else:
		linprog_bounds = bounds



	# Normalize constraints format as a list
	if not isinstance(constraints, list):
		constraints = [constraints]

	if verbose:
		printer = IterationPrinter(it = '4d', obj = '20.8e',  con= '20.8e', alpha = '8.3e')
		printer.print_header(it = 'iter', obj = "objective fun.", con = 'constraint violation',  alpha = 'step length')


	x = np.copy(x0)
	fx = fun(x)
	con_violation = 0
	for con in constraints:
		cx = con.fun(x)
		con_violation += np.sum(np.maximum(con.lb - cx, 0))
		con_violation += np.sum(np.maximum(cx - con.ub, 0))


	if verbose:
		printer.print_iter(it = 0, obj = fx, con = con_violation)
	mu = 1
	it = 0
	stop = False

	while it <= maxiter and not stop:
		it += 1
		
		assert False
