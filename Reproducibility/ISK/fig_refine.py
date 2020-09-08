import numpy as np
from polyrat import *
from pgf import PGF


data = []

# Absolute value test problem
M = int(2e5)
X = np.linspace(-1,1,M).reshape(-1,1)
y = np.abs(X).flatten()

data += [['abs', X, y]]

# Tangent
M = 1000
X = np.exp(1j*np.linspace(0, 2*np.pi, M, endpoint = False)).reshape(-1,1)
y = np.tan(256*X.flatten())
data += [['tan256', X, y]]

# Range of parameters
mns = [(k,k) for k in range(2, 22, 1)]

for name, X, y in data:

	err_isk = np.zeros(len(mns))
	err_iskr = np.zeros(len(mns))

	for k, (m, n) in enumerate(mns):
		
		print(f'\n======== SK Rebase ({m},{n}) =======\n')
		sk = SKRationalApproximation(m,n, verbose = True, rebase = True, refine = False, maxiter = 10)
		sk.fit(X,y)

		err_isk[k] = np.linalg.norm(sk(X) - y, 2)
		
		sk.refine(X, y)
		err_iskr[k] = np.linalg.norm(sk(X) - y, 2)

		print(f"original {err_isk[k]:10.3e}; improved {err_iskr[k]:10.3e}")


	pgf = PGF()
	pgf.add('m', [mn[0] for mn in mns])
	pgf.add('n', [mn[1] for mn in mns])
	pgf.add('err_isk', err_isk)
	pgf.add('err_iskr', err_iskr)

	pgf.write(f'data/fig_refine_{name}.dat')
