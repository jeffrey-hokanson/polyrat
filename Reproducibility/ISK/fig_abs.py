import numpy as np
from polyrat import *
from pgf import PGF

M = int(2e5)
X = np.linspace(-1,1,M).reshape(-1,1)
y = np.abs(X).flatten()

opts = [ 
	['legendre', dict(rebase = False, refine = False, Basis = LegendrePolynomialBasis)],
	['arnoldi', dict(rebase = False, refine = False, Basis = ArnoldiPolynomialBasis)],
	['arnoldi_rebase', dict(rebase = True, refine = False)],
	]

for name, kwargs in opts:
	# as abs is symmetric, we only use even powers	
	for m,n in [ (6,6), (10,10),  (14,14), (18,18)]: 

		print(f'\n======== {name}: ({m},{n}) =======\n')
		sk = SKRationalApproximation(m, n, verbose = True, **kwargs)
		sk.fit(X, y)

		pgf = PGF()
		pgf.add('cond', [h['cond'] for h in sk.hist])
		pgf.add('err', [np.linalg.norm(h['fit'] - y, 2) for h in sk.hist])
		delta_fit = [np.linalg.norm(sk.hist[0]['fit'], 2)] 
		delta_fit += [np.linalg.norm(sk.hist[i+1]['fit'] - sk.hist[i]['fit'],2) for i in range(len(sk.hist)-1)]
		pgf.add('delta_fit', delta_fit)
		pgf.write(f'data/fig_abs_{name}_{m}_{n}.dat')

