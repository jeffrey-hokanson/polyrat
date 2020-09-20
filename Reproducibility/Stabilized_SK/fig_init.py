import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from polyrat import *
from pgf import PGF


X = np.linspace(-1,1,int(2e5)).reshape(-1,1)
y = np.abs(X).flatten()
names = ['opt', 'vf', 'ssk']

for degree in [6,10, 14, 18]:

	vf_norms = []
	opt_norms = []
	sk_norms = []
	for seed in range(20):
		print(f"-----{seed:2d}------")
		np.random.seed(seed + 42*degree)		# Tweak the 
		xi = np.linspace(-1,1, degree+1)
		yi = np.random.randn(len(xi))
		p0 = LagrangePolynomialInterpolant(xi.reshape(-1,1), np.random.randn(len(xi)))
		q0 = LagrangePolynomialInterpolant(xi.reshape(-1,1), np.random.randn(len(xi)))
		
		
		# Construct ratio polynomial
		p = PolynomialApproximation(degree, Basis = ArnoldiPolynomialBasis)
		p.fit(X, p0(X)) 
		q = PolynomialApproximation(degree, Basis = ArnoldiPolynomialBasis)
		q.fit(X, q0(X)) 
		rat = RationalRatio(p,q)
		assert np.all(np.isclose(rat(X), p0(X)/q0(X)))

		#rat.refine(X, y, verbose = 2)
		opt_norms.append(np.linalg.norm(rat(X) - y))
		print(f"Optimization: {opt_norms[-1]:10.3e}")


		# Vector fitting
		vf = VectorFittingRationalApproximation(degree, degree, poles0 = q0.roots())
		vf.fit(X, y)
		vf_norms.append(np.linalg.norm(vf(X) - y))
		print(f"Vector Fitting: {vf_norms[-1]:10.3e}")
		
		# Stabilized SK
		sk = SKRationalApproximation(degree, degree, refine = False, rebase = True)
		sk.fit(X, y, denom0 = q0(X)) 	
		sk_norms.append(np.linalg.norm(sk(X) - y))
		print(f"Stabilized SK: {sk_norms[-1]:10.3e}")

	plt.close()
	ax = sns.swarmplot(data = [np.log10(d) for d in [opt_norms, vf_norms, sk_norms]])
	ax.set_yscale('log')

	for coll, name in zip(ax.collections, names):
		xx, yy = np.array(coll.get_offsets()).T
		pgf = PGF()
		pgf.add('x', xx)
		pgf.add('y', yy)
		if name != 'opt':
			pgf.write(f'data/fig_init_{degree:02d}_{name}.dat')
