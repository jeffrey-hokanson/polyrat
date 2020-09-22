import numpy as np
from scipy.linalg import block_diag
import tqdm
import scipy.io
from polyrat import *
from pgf import PGF, save_contour

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
	



import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,1)

# P-AAA

paaa = ParametricAAARationalApproximation(maxiter= 9)
paaa.fit(X, y)
err = np.abs(paaa(Xtest) - ytest).reshape(300,1000)

cs = ax[0].contourf(Z.imag, S.real, np.log10(err), levels = np.arange(-8,2)) 
save_contour('data/fig_penzl_contour_paaa.dat', cs, fmt = 'prepared')

Xi = paaa.interpolation_points
pgf = PGF()
pgf.add('z', Xi[:,0].imag)
pgf.add('s', Xi[:,1].real)
pgf.write('data/fig_penzl_paaa_interp.dat')

# Stabilized SK
num_degree = paaa.num_degree
denom_degree = paaa.denom_degree
ssk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 20)
ssk.fit(X, y)

err = np.abs(ssk(Xtest) - ytest).reshape(300,1000)

cs = ax[1].contourf(Z.imag, S.real, np.log10(err), levels = np.arange(-8,2)) 
save_contour('data/fig_penzl_contour_ssk.dat', cs, fmt = 'prepared')

# Linearized Rat. Approx.

lra = LinearizedRationalApproximation(num_degree, denom_degree)
lra.fit(X, y)

err = np.abs(lra(Xtest) - ytest).reshape(300,1000)

print("error on training set LRA", np.linalg.norm(lra(X) - y))

cs = ax[2].contourf(Z.imag, S.real, np.log10(err), levels = np.arange(-8,2)) 
save_contour('data/fig_penzl_contour_lra.dat', cs, fmt = 'prepared')

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[2].set_xscale('log')
fig.colorbar(cs)
plt.show()



