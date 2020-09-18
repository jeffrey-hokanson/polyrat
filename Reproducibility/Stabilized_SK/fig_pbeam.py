import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from polyrat import *


# load data
dat = scipy.io.loadmat('pbeam.dat')
FRF = dat['FRF']
z = dat['s'].flatten()
param = dat['par_p'].flatten()

# Convert into format [freq, s]
P, Z = np.meshgrid(param, z)
X = np.vstack([Z.flatten(), P.flatten()]).T
y = FRF.flatten()
# Include symmetric components
if False:
	I = (X[:,0] == 0.)
	X = np.vstack([X, X[~I,:].conj()]) 
	y = np.hstack([y, y[~I].conj()])

print(len(set(X[:,0])))

# Split into training
I = np.zeros(X.shape[0], dtype = np.bool)
for k in [np.argwhere(np.isclose(param, pp)) for pp in [0.2, 0.4, 1.]]:
	I = I | (X[:,1] == param[k]).flatten()
Xtrain = X[I,:]
ytrain = y[I]

paaa = ParametricAAARationalApproximation(tol = 1e-7, maxiter = 5)
paaa.fit(Xtrain, ytrain)

num_degree = paaa.num_degree
denom_degree = paaa.denom_degree
#num_degree = (11,1)
#denom_degree = (11,1)
ssk = SKRationalApproximation(num_degree, denom_degree, refine = False)
ssk.fit(Xtrain, ytrain)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)

err = np.abs(paaa(X) - y).reshape(len(z), len(param))
cs = ax[0].contourf(np.abs(P), np.abs(Z), np.log10(err), levels = np.arange(-10,0))


err = np.abs(ssk(X) - y).reshape(len(z), len(param))
ax[1].contourf(np.abs(P), np.abs(Z), np.log10(err), levels = np.arange(-10, 0))
fig.colorbar(cs)
plt.show()
