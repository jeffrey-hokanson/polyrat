import warnings
import numpy as np
from scipy.linalg import eig, eigvals, hessenberg
from .basis import PolynomialBasis
from .polynomial import Polynomial

try:
	from funtools import cached_property
except ImportError:
	from backports.cached_property import cached_property


def lagrange_roots(nodes, weights, coef, deflation = True):
	r""" Compute the roots of a Lagrange polynomial

	This implements the deflation algorithm from [LC14]_
	to compute the roots of a Lagrange polynomial 
	without changing basis and to high accuraccy.


	Parameters
	----------
	nodes: numpy.array (n,)
		Nodes :math:`x_j` where the polynomial value is defined.
	weights: numpy.array (n,)
		The barycentric weights :math:`\prod_{k\ne j} (\xi_j - \xi_k)^{-1}`
	coef: numpy.array (n,)
		The coeffients :math:`f_j` defining the value of the polynomial at each point;
		i.e., :math:`p(\xi_j) = f_j`.
	deflation: bool
		In the standard formulation to compute these roots
		we solve a generalized eigenvalue problem (GEP)
		which has two infinite poles.
		If True, we explicitly remove these infinite poles
		by shrinking the matrices;
		if False we do not.
	
	Returns
	-------
	roots: :class:`~numpy:numpy.ndarray`
		Roots of the polynomial.
	"""
	n = len(nodes)
	assert (n == len(weights)) and  (n == len(coef)), "Dimensions of nodes, weights, and coef should be the same"

	
	# Build the RHS of the generalized eigenvalue problem
	C0 = np.zeros((n+1, n+1), dtype=np.complex)
	C0[1:n+1,1:n+1] = np.diag(nodes)

	# LHS for generalized eigenvalue problem	
	C1 = np.eye(n+1, dtype=np.complex)
	C1[0, 0] = 0

	# scaling
	coef = coef / np.linalg.norm(coef)
	weights = weights / np.linalg.norm(weights)

	C0[0,1:n+1] = coef
	C0[1:n+1,0] = weights


	# balancing [LC14, eq. 29]
	coef0 = np.copy(coef)
	weights0 = np.copy(weights)
	s = np.array([1.]+[np.sqrt(np.abs(wj/aj)) if np.abs(aj) > 0 else 1 for (wj, aj) in zip(weights, coef)])
	C0 = np.diag(1/s) @ C0 @ np.diag(s)

	# Apply a rotation to make the first weight real
	angle = np.angle(C0[1,0])
	if np.isfinite(angle):
		C0[1:,0] *= np.exp(-1j*angle)
	else:
		print("Rotation failed", angle)
		deflation = False

	if deflation:
		#C0[1,0] must be real for Householder to reflect correctly
		assert np.abs(C0[1,0].imag) < 1e-10, "C0[1,0]: %g + I %g" % (C0[1,0].real, C0[1,0].imag)
		#Householder Reflector
		u = np.copy(C0[1:,0]) # = w scaled
		u[0] += np.linalg.norm(C0[1:,0]) # (w) scaled
		H = np.eye(n, dtype=complex) - 2 * np.outer(u,u.conjugate())/(np.linalg.norm(u)**2)
		G2 = np.zeros((n+1, n+1), dtype=complex)
		G2[0,0] = 1
		G2[1:,1:] = H
		C0 = G2 @ C0 @ G2
		C1 = G2 @ C1 @ G2
		H1, P1 = hessenberg(C0[1:,1:], calc_q=True, overwrite_a = False)
		G3 = np.zeros((n+1, n+1), dtype=complex)
		G3[0,0] = 1
		G3[1:,1:] = P1.T.conjugate()
		G4 = np.eye(n+1, dtype=complex)
		G4[0:2,0:2] = [[0,1],[1,0]]
		H1 = G4.dot(G3.dot(C0.dot(G3.T.conjugate())))[1:,1:]
		B1 = G4.dot(G3.dot(C1.dot(G3.T.conjugate())))[1:,1:]

		# Givens Rotation
		G5 = np.eye(n, dtype=complex)
		a = H1[0,0]
		b = H1[1,0]
		c = a / np.sqrt(a**2 + b**2)
		s = b / np.sqrt(a**2 + b**2)
		G5[0:2,0:2] = [[c.conjugate(), s.conjugate()],[-s,c]]

		H2 = G5.dot(H1)[1:,1:]
		B2 = G5.dot(B1)[1:,1:]
		try:
			return eigvals(H2, B2)
		except np.linalg.linalg.LinAlgError as e:
			raise e 
	else:
		# Compute the eigenvalues
		# As this eigenvalue problem has a double root at infinity, we ignore the division by zero warning
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", message='divide by zero encountered in true_divide',
									category=RuntimeWarning)
			ew = eigvals(C0, C1, overwrite_a=False)
		ew = ew[np.isfinite(ew).flatten()]
		assert len(ew) == len(coef) - 1, "Error: too many infinite eigenvalues encountered"

		return ew



def lagrange_vandermonde(nodes, weights, X):
	r""" Build the Vandermonde matrix associated with 
	"""
	x = X.flatten()
	assert len(x) == len(X), "Input must be one dimensional"

	with np.errstate(divide = 'ignore', invalid = 'ignore'):
		# The columns of the Vandermonde matrix 
		V = np.array([w/(x - n) for n, w in zip(nodes, weights)]).T
		for row in np.argwhere(~np.all(np.isfinite(V), axis = 1)):
			V[row] = 0
			V[row, np.argmin(np.abs(x[row] - nodes)).flatten()] = 1.
		
		denom = np.sum(V, axis = 1)
		V /= denom[:, None]
		
	
	return V


class LagrangePolynomialBasis(PolynomialBasis):
	r""" Constructs a Lagrange polynomial basis in barycentric form.


	Here we construct a univariate polynomial in barycentric form
	as described in [BT04]_:
	
	.. math::

		p(x) = \sum_{j=1}^d \frac{ f_j w_j (x - \xi_j)^{-1}}{w_j (x - \xi_j)^{-1}},
		\quad w_j := \prod_{k\ne j} (\xi_j - \xi_k)^{-1}

	such that :math:`p(\xi_j) = f_j`.

	Parameters
	----------
	nodes: array_like
		List of the nodes :math:`\xi_j` specifying the basis.

	"""
	def __init__(self, nodes):
		self.nodes = np.array(nodes).flatten()
		assert len(nodes) == len(self.nodes), "Input must be one-dimensional"

		# BT 04, eq. (3.2)
		# w[j] = prod_{j\ne k} 1./(node[j] - node[k])
		self.weights = 1./np.array([
			np.prod(self.nodes[k] - self.nodes[0:k]) * np.prod(self.nodes[k] - self.nodes[k+1:]) for k in range(len(self.nodes))
			])


	@cached_property
	def vandermonde_X(self):
		return np.eye(len(nodes))

	def vandermonde(self, X):
		return lagrange_vandermonde(self.nodes, self.weights, X)

	def vandermonde_derivative(self, X):
		raise NotImplementedError
	
	def roots(self, coef, deflation = True):
		return lagrange_roots(self.nodes, self.weights, coef, deflation = deflation)


class LagrangePolynomialInterpolant(Polynomial):
	def __init__(self, X, y):
		self.basis = LagrangePolynomialBasis(X)
		self.coef = np.copy(y)
	
	@property
	def nodes(self):
		return self.basis.nodes


