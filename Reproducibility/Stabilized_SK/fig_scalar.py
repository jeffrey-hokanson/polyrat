import numpy as np
from polyrat import *
from pgf import PGF
from scipy.special import jv, erf, gamma, rgamma, sinc
import os
import scipy.io
dir_path = os.path.dirname(os.path.realpath(__file__))

data = []

# Two real-valued input problems

# Absolute value test problem
M = int(2e5)
X = np.linspace(-1,1,M).reshape(-1,1)
y = np.abs(X).flatten()
data += [['abs', X, y]]

# Exponential (NST18: Sec. 6.8)
M = 2000
X = -np.logspace(-3, 4, M).reshape(-1,1)
y = np.exp(X.flatten())
data += [['exp', X, y]]


# Error function
M = int(1e4)
X = np.hstack([-np.logspace(-4, 1, M)[::-1], np.logspace(-4, 1, M)]).reshape(-1,1)
y = erf(X.flatten())
data += [['erf', X, y]]

# Sine function
M = int(1e4)
X = np.linspace(-2*np.pi, 2*np.pi, M).reshape(-1,1)
y = np.sin(X.flatten())
data += [['sine', X, y]]

# Sinc function
M = int(1e4)
X = np.linspace(0, 10, M).reshape(-1,1)
y = sinc(X.flatten())
data += [['sinc', X, y]]


# Gamma function
M = int(1e4)
X = np.linspace(-1000, 5, M).reshape(-1,1)
y = gamma(X.flatten())
I = np.isfinite(y).flatten()
X = X[I]
y = y[I]
data += [['gamma', X, y]]

M = int(1e4)
X = np.linspace(-10, 10, M).reshape(-1,1)
y = rgamma(X.flatten())
data += [['rgamma', X, y]]

# Two complex-valued input problems

# Tangent
M = 1000
X = np.exp(1j*np.linspace(0, 2*np.pi, M, endpoint = False)).reshape(-1,1)
y = np.tan(256*X.flatten())
data += [['tan256', X, y]]

# Bessel function (NST18 Fig 6.5)
# This converges too fast
M = 2000
np.random.seed(0)
X = 10*np.random.rand(M) + 2j*(np.random.rand(M) - 0.5)
y = 1./jv(0, X)
data += [['bessel', X.reshape(-1,1), y]]

# Bessel function modified to a real domain
M = int(1e4)
#np.random.seed(0)
#X = 100*np.random.rand(M)
X = np.linspace(0, 100, M)
y = jv(0, X)
I = np.isfinite(y)
X = X[I]
y = y[I]
data += [['real_bessel', X.reshape(-1,1), y]]

# Log example: NST18 Fig 6.2;
# note rate is independent of offset, so shinking does not harm convergence
M = int(2e3)
X = np.exp(1j*np.linspace(0, 2*np.pi, M, endpoint = False)).reshape(-1,1)
y = np.log(1.1 - X.flatten())
data += [['log11', X, y]]

# Beam example
fname = os.path.join(dir_path, 'beam.mat')
d = scipy.io.loadmat(fname)
A = d['A']
B = d['B']
C = d['C']

X = 1j*np.logspace(-2, 2, 500)
X = np.hstack([X , X.conj()])
y = np.array([ C @ np.linalg.solve(z*np.eye(A.shape[0]) - A, B)  for z in X]).flatten()
data += [['beam', X.reshape(-1,1), y]]

# Pick which experiments to run
data = [d for d in data if d[0] in ['abs', 'beam', 'tan256', 'exp']]

# Range of parameters
mns = [(k,k) for k in range(2, 51, 1)]

for name, X, y in data:

	err_isk = np.zeros(len(mns))
	err_iskr = np.zeros(len(mns))
	err_aaa = np.zeros(len(mns))
	err_lra = np.zeros(len(mns))
	err_vf = np.zeros(len(mns))

	for k, (m, n) in enumerate(mns):
	
		print(f'\n======== AAA ({m},{m}) | {name} =======\n')
		aaa = AAARationalApproximation(degree = m)
		aaa.fit(X, y)
		err_aaa[k] = np.linalg.norm(aaa(X) - y, 2)/np.linalg.norm(y,2)
		

		print(f'\n======== Linearized ({m},{n}) | {name} =======\n')
		lra = LinearizedRationalApproximation(m,n)
		lra.fit(X, y)
		err_lra[k] = np.linalg.norm(lra(X) - y, 2)/np.linalg.norm(y,2)
		print(f"error {err_lra[k]:20.15e}")
	
		print(f'\n======== SK Rebase ({m},{n}) | {name} =======\n')
		sk = SKRationalApproximation(m,n, verbose = True, rebase = True, refine = False, maxiter = 20, xtol = 0)
		sk.fit(X,y)

		err_isk[k] = np.linalg.norm(sk(X) - y, 2)/np.linalg.norm(y, 2)
		
		sk.refine(X, y, verbose = 2, method = 'lm')
		err_iskr[k] = np.linalg.norm(sk(X) - y, 2)/np.linalg.norm(y, 2)

		print(f"original {err_isk[k]:10.3e}; improved {err_iskr[k]:10.3e}")
		

		print(f'\n======== VectorFitting ({m},{m}) | {name} =======\n')
		vf = VectorFittingRationalApproximation(num_degree = m, denom_degree = m)
		vf.fit(X, y)
		err_vf[k] = np.linalg.norm(vf(X) - y, 2)/np.linalg.norm(y,2)


		pgf = PGF()
		pgf.add('m', [mn[0] for mn in mns])
		pgf.add('n', [mn[1] for mn in mns])
		pgf.add('err_isk', err_isk)
		pgf.add('err_iskr', err_iskr)
		pgf.add('err_aaa', err_aaa)
		pgf.add('err_lra', err_lra)
		pgf.add('err_vf', err_vf)

		pgf.write(f'data/fig_scalar_{name}.dat')
