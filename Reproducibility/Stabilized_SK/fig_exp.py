import numpy as np
import matplotlib.pyplot as plt
from polyrat import *

np.random.seed(0)
X = np.random.uniform(-1,1, size = (1000,2))
y = np.exp(X[:,0]*X[:,1])/( (X[:,0]**2 - 1.2**2) * (X[:,1]**2 - 1.2**2))

lra = LinearizedRationalApproximation(20, 20)
lra.fit(X, y)

ssk = SKRationalApproximation(20, 20, refine = False)
ssk.fit(X, y)

X = np.linspace(-1.5,1.5,200)
Y = X.copy()
X, Y = np.meshgrid(X, Y)

XX = np.vstack([X.flatten(), Y.flatten()]).T
denom1 = lra.denominator(XX)
denom1 /= np.max(denom1)
denom2 = ssk.denominator(XX)
denom2 /= np.max(denom2)

fig, ax = plt.subplots(1,2)
ax[0].contour(X, Y, denom1.reshape(X.shape), levels = np.linspace(-1,1,11))
ax[0].contour(X, Y, denom1.reshape(X.shape), levels = [0], colors = 'red')
ax[1].contour(X, Y, denom2.reshape(X.shape), levels = np.linspace(-1,1,11))
ax[1].contour(X, Y, denom2.reshape(X.shape), levels = [0], colors = 'red')

plt.show()
