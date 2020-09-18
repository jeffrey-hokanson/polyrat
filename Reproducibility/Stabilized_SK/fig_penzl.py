import numpy as np
from scipy.linalg import block_diag
import tqdm
import scipy.io
from polyrat import *
from pgf import PGF

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
		#H[i] = c.T @ np.linalg.solve(z*np.eye(1006) - A, b)
		H[i] += c[0:2].T @ np.linalg.solve(z*np.eye(2) - A1, b[0:2])
		H[i] += c[2:4].T @ np.linalg.solve(z*np.eye(2) - A2, b[2:4])
		H[i] += c[4:6].T @ np.linalg.solve(z*np.eye(2) - A3, b[4:6])
		H[i] += c[6:].T @ (1./(z - np.diag(A4))).reshape(-1,1) 

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
ssk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 20)
ssk.fit(X, y)

print("error on training set P-AAA", np.linalg.norm(paaa(X) - y))
print("error on training set S-SK", np.linalg.norm(ssk(X) - y))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)

err = np.abs(paaa(Xtest) - ytest).reshape(300,1000)

pgf = PGF()
pgf.add('z', np.log10(Xtest[:,0].imag))
pgf.add('s', Xtest[:,1].real)
pgf.add('err', np.log10(err.flatten()))
pgf.write('data/fig_penzl_paaa.dat')

# small
pgf = PGF()
pgf.add('z', np.log10(X[:,0].imag))
pgf.add('s', X[:,1].real)
pgf.add('err', np.log10(np.abs(paaa(X) - y)+1e-50))
pgf.write('data/fig_penzl_paaa_small.dat')


ax[0].set_yscale('log')
cs = ax[0].contourf(np.abs(S), np.abs(Z), np.log10(err), levels = np.arange(-8,0))


Xi = paaa.interpolation_points
pgf = PGF()
pgf.add('z', np.log10(Xi[:,0].imag))
pgf.add('s', Xi[:,1].real)
pgf.write('data/fig_penzl_paaa_interp.dat')

ax[0].plot(np.abs(Xi[:,1]), np.abs(Xi[:,0]), 'r.')

err = np.abs(ssk(Xtest) - ytest).reshape(300,1000)

pgf = PGF()
pgf.add('z', np.log10(Xtest[:,0].imag))
pgf.add('s', Xtest[:,1].real)
pgf.add('err', np.log10(err.flatten()))
pgf.write('data/fig_penzl_ssk.dat')

pgf = PGF()
pgf.add('z', np.log10(X[:,0].imag))
pgf.add('s', X[:,1].real)
pgf.add('err', np.log10(np.abs(ssk(X) - y)+1e-50))
pgf.write('data/fig_penzl_ssk_small.dat')

ax[1].set_yscale('log')
ax[1].contourf(np.abs(S), np.abs(Z), np.log10(err), levels = np.arange(-8,0))
fig.colorbar(cs)
plt.show()



