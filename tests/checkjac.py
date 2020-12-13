import numpy as np

def check_jacobian(x, residual, jacobian, h = 2e-7, relative = False):
	n = x.shape[0]

	J = jacobian(x)

	err = np.zeros(n)

	print("Jacobian condition number", np.linalg.cond(J))

	for i in range(n):
		ei = np.zeros(x.shape, dtype = np.float)
		ei[i]= 1.		
			
		x1 = x + ei * h
		x2 = x - ei * h
		r1 = residual(x1) 
		r2 = residual(x2)
		Ji_est = (r1 - r2)/(2*h)
		
		err[i] = np.linalg.norm(Ji_est - J[:,i],2)
		print(f" {i:3d} : error {err[i]:8.2e}, relative {err[i]/np.linalg.norm(J[:,i],2):8.2e}")
		if relative:
			err[i] /= np.linalg.norm(J[:,i],2)

	return np.max(err)
