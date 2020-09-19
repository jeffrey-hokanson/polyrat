import numpy as np
import scipy.io
from polyrat import *
from pgf import PGF


# Load data
dat = scipy.io.loadmat('penzl3.dat')	
X = dat['X']
y = dat['y'].flatten()


n1, n2, n3 = [],[],[]
d1, d2, d3 = [],[],[]
err = []
for d in [5, 6, 7, 8, 9, 10]:
	num_degree = [d-1, 1,3]
	denom_degree = [d, 2, 4]

	ssk = SKRationalApproximation(num_degree, denom_degree, refine = False, maxiter = 10)
	ssk.fit(X, y)

	n1.append(num_degree[0])	
	n2.append(num_degree[1])	
	n3.append(num_degree[2])	
	
	d1.append(denom_degree[0])	
	d2.append(denom_degree[1])	
	d3.append(denom_degree[2])	
	err.append(np.linalg.norm(ssk(X) - y))

	pgf = PGF()
	pgf.add('n1', n1)
	pgf.add('n2', n2)
	pgf.add('n3', n3)
	
	pgf.add('d1', d1)
	pgf.add('d2', d2)
	pgf.add('d3', d3)
	pgf.add('err', err)
	pgf.write('data/fig_penzl3_ssk.dat')
