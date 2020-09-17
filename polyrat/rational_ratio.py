r""" Tools to find a locally optimal rational approximation in ratio format

In the ratio format, we assume we have bases P and Q for the numerator and denominator 
"""

import numpy as np
import scipy.optimize
from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds
from .slp import sl1lp, slp

################################################################################
# Residual and jacobian  
################################################################################

def _rational_residual_real(x, P, Q, y):
	m = P.shape[1]
	n = Q.shape[1]
	Pa = P @ x[:m]
	Qb = Q @ x[-n:]
	return Pa/Qb - y 

def _rational_jacobian_real(x, P, Q):
	m = P.shape[1]
	n = Q.shape[1]
	Pa = P @ x[:m]
	Qb = Q @ x[-n:]
	return np.hstack([			
			np.multiply((1./Qb)[:,None], P),				
			np.multiply(-(Pa/Qb**2)[:,None], Q),
		])
			

def _rational_residual_complex(x, P, Q, y):
	m = P.shape[1]
	n = Q.shape[1]
	a = x[:2*m].view(complex)
	b = x[-2*n:].view(complex)
	res = (P @ a)/(Q @ b) - y
	return res.view(float)


def _rational_jacobian_complex(x, P, Q):
	m = P.shape[1]
	n = Q.shape[1]
	
	a = x[:2*m].view(complex)
	b = x[-2*n:].view(complex)
	Pa = P @ a
	Qb = Q @ b
	J = np.hstack([			
			np.multiply((1./Qb)[:,None], P),				
			np.multiply(-(Pa/Qb**2)[:,None], Q),
		])
	JRI = np.zeros((J.shape[0]*2, J.shape[1]*2), dtype = np.float)
	JRI[0::2,0::2] = J.real
	JRI[1::2,1::2] = J.real
	JRI[0::2,1::2] = -J.imag
	JRI[1::2,0::2] = J.imag
	return JRI


################################################################################
# 2-norm optimization proceedures
################################################################################

def _rational_ratio_optimize_pin_2norm_real(y, P, Q, a0, b0, **kwargs):
	m, n = P.shape[1], Q.shape[1]
	k = np.argmax(np.abs(b0))
	x0 = np.hstack([a0, b0[:k], b0[k+1:]])
	xfun = lambda x: np.hstack([x[:m+k], b0[k], x[m+k:]])
	
	mask = np.ones(m+n, dtype = np.bool)
	mask[m+k] = 0
	res = lambda x: _rational_residual_real(xfun(x), P, Q, y)
	jac = lambda x: _rational_jacobian_real(xfun(x), P, Q)[:,mask]

	result = scipy.optimize.least_squares(res, x0, jac, **kwargs)
	x = xfun(result.x)
	return x[:m], x[m:]


def _rational_ratio_optimize_pin_2norm_complex(y, P, Q, a0, b0, **kwargs):
	m, n = P.shape[1], Q.shape[1]
	k = np.argmax(np.abs(b0))
	x0 = np.hstack([a0, b0[:k], b0[k+1:]])
	xfun = lambda x: np.hstack([x[:2*(m+k)], b0[k].real, b0[k].imag, x[2*(m+k):]])
	
	mask = np.ones(2*(m+n), dtype = np.bool)
	mask[2*(m+k):2*(m+k)+2] = 0
	res = lambda x: _rational_residual_complex(xfun(x), P, Q, y)
	jac = lambda x: _rational_jacobian_complex(xfun(x), P, Q)[:,mask]

	result = scipy.optimize.least_squares(res, x0.view(float), jac, **kwargs)
	x = xfun(result.x)
	return x[:2*m].view(complex), x[2*m:].view(complex)
	

################################################################################
# sup-norm optimization utilities
################################################################################

# setup the objective for the constraint
def _rational_residual_squared_abs_complex(x, P, Q, y):
	r = _rational_residual_complex(x, P, Q, y)
	return r[::2]**2 + r[1::2]**2

def _rational_jacobian_squared_abs_complex(x, P, Q, y):
	r = _rational_residual_complex(x, P, Q, y)
	J = _rational_jacobian_complex(x, P, Q)
	Jt = np.zeros((r.shape[0]//2, x.shape[0]))
	Jt += 2*np.diag(r[0::2]) @ J[0::2,:]
	Jt += 2*np.diag(r[1::2]) @ J[1::2,:]
	return Jt	

def _rational_ratio_inf_complex(y, P, Q, a0, b0):
	r"""

	Solving as 

	min_{x, t} 0.5 * t**2
	st.        Re[f_j(x)]^2 + Im[f_j(x)]^2 - t^2 <= 0

	"""

	# Introduce a slack variable t to represent the maximum
	fun = lambda xt: 0.5*xt[-1]**2
	def grad(xt):
		g = np.zeros(xt.shape)
		g[-1] = xt[-1]
		return g

	def hess(xt):
		H = np.zeros( (len(xt), len(xt)) )
		H[-1,-1] = 1
		return H

	def con(xt):
		x = xt[:-1]
		t = xt[-1]
		r =  _rational_residual_squared_abs_complex(x, P, Q, y) - t**2
		return r

	def con_jac(xt):
		x = xt[:-1]
		t = xt[-1]
		J = _rational_jacobian_squared_abs_complex(x, P, Q, y)
		Jt = np.hstack([J, -2*t*np.ones((J.shape[0], 1)) ])
		return Jt

	lb = -np.inf*np.ones(P.shape[0])
	ub = np.zeros(P.shape[0])
	constraint = NonlinearConstraint(con, lb, ub, jac = con_jac)


	t0 = np.max(np.abs( (P @ a0)/(Q @ b0) - y))
	xt0 = np.hstack([a0.view(float), b0.view(float), t0])
	
	lb_var = -np.inf*np.ones(xt0.shape)
	lb_var[-1] = 0
	ub_var = np.inf*np.ones(xt0.shape)
	#print("t0", t0)

	# TODO: Replace with a call to SLP
	res = scipy.optimize.minimize(fun, xt0, jac = grad, constraints = constraint, method = slp,
		bounds = Bounds(lb_var, ub_var), options = dict(maxiter = 50))
	xt = res.x
	x = xt[:-1]
	a = x[:2*P.shape[1]].view(complex)
	b = x[-2*Q.shape[1]:].view(complex)
	#print("t", xt[-1])
	#print("con", np.max(con(xt)))
	
	return a, b

def rational_ratio_optimize(y, P, Q, a0, b0, norm = 2, **kwargs):
	r""" Find a locally optimal rational approximation in ratio form



	Parameters
	----------
	y
	P
	Q
	a0
	b0
	norm: [1, 2, np.inf]
		Norm in which to construct the approximation
	**kwargs
		Additional arguments to be passed to the optimization solver
	"""
	isreal = all([np.all(np.isreal(x)) for x in [y, P, Q, a0, b0]])

	m = P.shape[1]
	n = Q.shape[1]	
	x0 = np.hstack([a0, b0])


	if not isreal:
		x0 = x0.astype(np.complex)
		P = P.astype(np.complex)
		Q = Q.astype(np.complex)
		y = y.astype(np.complex)	
	
	if isreal and norm == 2:
		return _rational_ratio_optimize_pin_2norm_real(y, P, Q, a0, b0, **kwargs)
			
	elif (not isreal) and norm == 2:
		return _rational_ratio_optimize_pin_2norm_complex(y, P, Q, a0, b0, **kwargs)

	else:
		mess = f"The combination of norm={norm} and "
		if isreal:
			mess +="real data "
		else:
			mess +="complex data "
		mess+= "is not implemented"
		raise NotImplementedError(mess)
 
		


