import numpy as np
import polyrat
import matplotlib.pyplot as plt

X = np.linspace(-1,1, 1000).reshape(-1,1)
y = np.tan(2*np.pi*X.flatten())

num_degree, denom_degree = 10, 10
rat = polyrat.StabilizedSKRationalApproximation(num_degree, denom_degree)
rat.fit(X, y)

fig, ax = plt.subplots()
ax.plot(X, y, '.', label = r'$\tan(2\pi x)$ training data')
ax.plot(X, rat(X), '-', label = r'rational approximation')
ax.set_ylim(-10, 10)
ax.set_title(r'Degree (10,10) Rational Approximation of $\tan(2\pi x)$')
ax.set_xlabel('x'); ax.set_ylabel(r'$\tan(2\pi x)$')
ax.legend(loc = 'lower right')
fig.savefig('tan.png', dpi = 300)
plt.show()

