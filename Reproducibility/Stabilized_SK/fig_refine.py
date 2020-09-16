import numpy as np
from polyrat import *
from polyrat.rational_ratio import _rational_jacobian_complex, _rational_jacobian_real
from pgf import PGF


data = []

# Absolute value test problem
M = int(2e3)
X = np.linspace(-1,1,M).reshape(-1,1)
y = np.abs(X).flatten()

#data += [['abs', X, y]]

# Tangent
M = 10000
X = np.exp(1j*np.linspace(0, 2*np.pi, M, endpoint = False)).reshape(-1,1)
y = np.tan(1024*X.flatten())
data += [['tan1024', X, y]]

# Range of parameters
mns = [(k,k) for k in range(2, 51, 1)]

for name, X, y in data:

	err_isk = np.zeros(len(mns))
	err_iskr = np.zeros(len(mns))
	delta_a = np.zeros(len(mns))
	delta_b = np.zeros(len(mns))
	conds = np.zeros(len(mns))

	for k, (m, n) in enumerate(mns):
		
		print(f'\n======== SK Rebase ({m},{n}) =======\n')
		sk = SKRationalApproximation(m,n, verbose = True, rebase = True, refine = False, maxiter = 10)
		sk.fit(X,y)
		
		rot = np.linalg.norm(sk.b)*sk.b[0]/np.abs(sk.b[0])
		a0, b0 = sk.a/rot, sk.b/rot

		err_isk[k] = np.linalg.norm(sk(X) - y, 2)/np.linalg.norm(y, 2)
		
		sk.refine(X, y, verbose = 2)
		err_iskr[k] = np.linalg.norm(sk(X) - y, 2)/np.linalg.norm(y, 2)

		print(f"original {err_isk[k]:10.3e}; improved {err_iskr[k]:10.3e}")
		
		
		rot = np.linalg.norm(sk.b)*sk.b[0]/np.abs(sk.b[0])
		a1, b1 = sk.a/rot, sk.b/rot
		
		delta_a[k] = np.linalg.norm(a0 - a1)
		delta_b[k] = np.linalg.norm(b0 - b1)
		J = _rational_jacobian_complex(np.hstack([a1,b1]).astype(np.complex).view(float),sk.P, sk.Q)
		s = np.linalg.svd(J, compute_uv = False)
		print("sing", s)
		conds[k] = s[0]/s[-3]
		pgf = PGF()
		pgf.add('m', [mn[0] for mn in mns])
		pgf.add('n', [mn[1] for mn in mns])
		pgf.add('err_isk', err_isk)
		pgf.add('err_iskr', err_iskr)
		pgf.add('delta_a', delta_a)
		pgf.add('delta_b', delta_b)
		pgf.add('cond', conds)

		pgf.write(f'data/fig_refine_{name}.dat')
