import numpy as np
from scipy.linalg import block_diag
import tqdm
import scipy.io
from polyrat import *
from pgf import PGF


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
		#H[i] = c.T @ np.linalg.solve(z*np.eye(1006) - A, b)
		H[i] += c[0:2].T @ np.linalg.solve(z*np.eye(2) - A1, b[0:2])
		H[i] += c[2:4].T @ np.linalg.solve(z*np.eye(2) - A2, b[2:4])
		H[i] += c[4:6].T @ np.linalg.solve(z*np.eye(2) - A3, b[4:6])
		H[i] += c[6:].T @ (1./(z - np.diag(A4))).reshape(-1,1) 

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
	

d1 = []
d2 = []
d3 = []
paaa_err = []
lra_err = []
ssk_err = []

for maxiter in range(7, 15):
	paaa = ParametricAAARationalApproximation(maxiter = maxiter)
	paaa.fit(X, y)
	paaa_err.append(np.linalg.norm(paaa(X) - y))

	num_degree = paaa.num_degree
	denom_degree = paaa.denom_degree

	d1.append(num_degree[0])
	d2.append(num_degree[1])
	d3.append(num_degree[2])


	lra = LinearizedRationalApproximation(num_degree, denom_degree)
	lra.fit(X, y)
	lra_err.append(np.linalg.norm(lra(X) - y))
	print(f"LRA error {lra_err[-1]:10.5e}")
	
	ssk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 10)
	ssk.fit(X, y)
	ssk_err.append(np.linalg.norm(ssk(X) - y))

	pgf = PGF()
	pgf.add('d1', d1)
	pgf.add('d2', d2)
	pgf.add('d3', d3)
	pgf.add('paaa_err', paaa_err)
	pgf.add('lra_err', paaa_err)
	pgf.add('ssk_err', ssk_err)
	pgf.write('data/fig_penzl3.dat')

	if num_degree[0]>= 12:
		break
