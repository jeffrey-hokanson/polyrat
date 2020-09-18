import numpy as np
from scipy.linalg import block_diag
import tqdm
import scipy.io
from polyrat import *



def penzl3(X):
	
	b = np.ones(1006)
	b[0:6] *= 10
	c = np.copy(b)

	A4 = -np.diag(np.arange(1, 1001))

	H = np.zeros(len(X), dtype = np.complex)
	for i, x in tqdm.tqdm(enumerate(X), total = len(X)):
		z = x[0]
		p1 = x[1]
		p2 = x[2]
		A1 = np.array([[-1,p1],[-p1,-1]])
		A2 = np.array([[-1, p2],[-p2,-1]])
		A3 = np.array([[-1,2*p2], [-2*p2,-1]])
		A = block_diag(A1, A2, A3, A4)
		H[i] = c.T @ np.linalg.solve(z*np.eye(1006) - A, b)

	return H

try:
	dat = scipy.io.loadmat('penzl3.dat')	
	X = dat['X']
	y = dat['y'].flatten()
except FileNotFoundError:
	z = 1j*np.logspace(-1, 3, 100)
	s1 = np.linspace(10, 100, 10)
	s2 = np.linspace(150, 250, 10)
	Z, S1, S2 = np.meshgrid(z, s1, s2)
	X = np.vstack([Z.flatten(), S1.flatten(), S2.flatten()]).T
	y = penzl3(X)
	
	scipy.io.savemat('penzl3.dat', {'X':X, 'y':y})
	

paaa = ParametricAAARationalApproximation()
paaa.fit(X, y)

num_degree = paaa.num_degree
denom_degree = paaa.denom_degree
ssk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 5)
ssk.fit(X, y)


