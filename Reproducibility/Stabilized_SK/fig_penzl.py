import numpy as np
from scipy.linalg import block_diag
import tqdm
import scipy.io
from polyrat import *


def penzl2(X):
	
	b = np.ones(1006)
	b[0:6] *= 10
	c = np.copy(b)

	A2 = np.array([[-1, 200],[-200,-1]])
	A3 = np.array([[-1, 400], [-400,-1]])
	A4 = -np.diag(np.arange(1, 1001))

	H = np.zeros(len(X), dtype = np.complex)
	for i, x in tqdm.tqdm(enumerate(X), total = len(X)):
		z = x[0]
		p = x[1]
		A1 = np.array([[-1,p],[-p,-1]])
		A = block_diag(A1, A2, A3, A4)
		H[i] = c.T @ np.linalg.solve(z*np.eye(1006) - A, b)

	return H

def penzl3(X):
	
	b = np.ones(1006)
	b[0:6] *= 10
	c = np.copy(b)

	A4 = -np.diag(np.arange(1, 1001))

	H = np.zeros(len(X), dtype = np.complex)
	for i, x in enumerate(X):
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
	dat = scipy.io.loadmat('penzl2.dat')	
	X = dat['X']
	y = dat['y'].flatten()
	Xtest = dat['Xtest']
	ytest = dat['ytest'].flatten()
	Z = Xtest[:,0].reshape(300,1000)
	S = Xtest[:,1].reshape(300,1000)
except FileNotFoundError:
	z = 1j*np.logspace(-1, 3, 100)
	s = np.linspace(10, 100, 30)
	Z, S = np.meshgrid(z, s)
	X = np.vstack([Z.flatten(), S.flatten()]).T
	y = penzl2(X)
	
	z = 1j*np.logspace(-1, 3, 1000)
	s = np.linspace(10, 100, 300)
	Z, S = np.meshgrid(z, s)
	Xtest = np.vstack([Z.flatten(), S.flatten()]).T
	ytest = penzl2(Xtest)
	scipy.io.savemat('penzl2.dat', {'X':X, 'y':y, 'Xtest':Xtest, 'ytest':ytest})
	

paaa = ParametricAAARationalApproximation(maxiter= 9)
paaa.fit(X, y)

num_degree = paaa.num_degree
denom_degree = paaa.denom_degree
ssk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 5)
ssk.fit(X, y)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)

err = np.abs(paaa(Xtest) - ytest).reshape(300,1000)
ax[0].set_yscale('log')
cs = ax[0].contourf(np.abs(S), np.abs(Z), np.log10(err), levels = np.arange(-10,1))

err = np.abs(ssk(Xtest) - ytest).reshape(300,1000)
ax[1].set_yscale('log')
ax[1].contourf(np.abs(S), np.abs(Z), np.log10(err), levels = np.arange(-10,1))
fig.colorbar(cs)
plt.show()



