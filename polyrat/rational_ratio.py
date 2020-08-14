r""" Tools to find a locally optimal rational approximation in ratio format

In the ratio format, we assume we have bases P and Q for the numerator and denominator 
"""

import numpy as np


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




def rational_ratio_optimize(y, P, Q, a0, b0, norm = 2):
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
	"""
	isreal = all([np.all(np.isreal(x)) for x in [y, P, Q, a0, b0]])

	m = P.shape[1]
	n = Q.shape[1]	
	x0 = np.hstack([a0, b0])
	
	if isreal and norm == 2:
		res = lambda x: _rational_residual_real(x, P, Q, y)
		jac = lambda x: _rational_jacobian_real(x, P, Q) 
		res = scipy.optimize.least_squares(res, x0, jac)
		a = res.x[:m]
		b = res.x[-n:]
		return a, b	
			
	elif (not isreal) and norm == 2:
		res = lambda x: _rational_residual_complex(x, P, Q, y)
		jac = lambda x: _rational_jacobian_complex(x, P, Q) 
		res = scipy.optimize.least_squares(res, x0.view(float), jac)
		a = res.x[:2*m].view(complex)
		b = res.x[-2*n:].view(complex)	
		return a,b

	else:
		mess = f"The combination of norm={norm} and "
		if isreal:
			mess +="real data "
		else:
			mess +="complex data "
		mess+= "is not implemented"
		raise NotImplementedError(mess)
 
		


