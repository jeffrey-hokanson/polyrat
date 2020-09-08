import numpy as np
from polyrat import *
from pgf import PGF
from scipy.special import jv

data = []

# Two real-valued input problems

# Absolute value test problem
M = int(2e3)
X = np.linspace(-1,1,M).reshape(-1,1)
y = np.abs(X).flatten()
data += [['abs', X, y]]

# Exponential (NST18: Sec. 6.8)
M = 4000
X = -np.logspace(-3, 4, M).reshape(-1,1)
y = np.exp(X.flatten())
data += [['exp', X, y]]

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

# Log example: NST18 Fig 6.2;
# note rate is independent of offset, so shinking does not harm convergence
M = int(2e3)
X = np.exp(1j*np.linspace(0, 2*np.pi, M, endpoint = False)).reshape(-1,1)
y = np.log(1.1 - X.flatten())
data += [['log11', X, y]]


data = data[2:3]

# Range of parameters
mns = [(k,k) for k in range(2, 51, 1)]



for name, X, y in data:

	err_isk = np.zeros(len(mns))
	err_iskr = np.zeros(len(mns))
	err_aaa = np.zeros(len(mns))

	for k, (m, n) in enumerate(mns):
	
		print(f'\n======== AAA ({m},{m}) | {name} =======\n')
		aaa = AAARationalApproximation(degree = m)
		aaa.fit(X, y)
		err_aaa[k] = np.linalg.norm(aaa(X) - y, 2)/np.linalg.norm(y,2)

	
		print(f'\n======== SK Rebase ({m},{n}) | {name} =======\n')
		sk = SKRationalApproximation(m,n, verbose = True, rebase = True, refine = False, maxiter = 10)
		sk.fit(X,y)

		err_isk[k] = np.linalg.norm(sk(X) - y, 2)/np.linalg.norm(y, 2)
		
		sk.refine(X, y)
		err_iskr[k] = np.linalg.norm(sk(X) - y, 2)/np.linalg.norm(y, 2)

		print(f"original {err_isk[k]:10.3e}; improved {err_iskr[k]:10.3e}")


	pgf = PGF()
	pgf.add('m', [mn[0] for mn in mns])
	pgf.add('n', [mn[1] for mn in mns])
	pgf.add('err_isk', err_isk)
	pgf.add('err_iskr', err_iskr)
	pgf.add('err_aaa', err_aaa)

	pgf.write(f'data/fig_scalar_{name}.dat')
