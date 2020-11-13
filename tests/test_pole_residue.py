import numpy as np
from polyrat import *
from polyrat.pole_residue import *


def test_residual_real():
	np.random.seed(0)
	M = 1000
	p, m = 1, 1	
	x = np.linspace(-1,1, M)
	Y = np.random.randn(M, 1)

	lam = np.array([-0.1, -0.2])
	a = np.array([1,-1])
	V = np.zeros((M,0))
	c = np.zeros((0))

	r = residual_jacobian_real(x, Y, lam, a, V, c, jacobian = False) 

	r_true = np.copy(Y)

	for i in range(Y.shape[0]):
		for idx in np.ndindex(Y.shape[1:]):
			for j in range(len(lam)):
				r_true[i, idx] -= a[j]/(x[i] - lam[j])
			for j in range(len(c)):
				r_true[i,idx] -= V[i,j]*c[j]

	print(r_true)	

if __name__ == '__main__':
	test_residual_real()

