import numpy as np
from polyrat import *



def test_paaa():
	r""" CG20x, subsec.~3.2.1 "Synthetic transfer function"
	"""	
	s = np.linspace(-1,1,21)
	p = np.linspace(0, 1, 21)
	S, P = np.meshgrid(s, p)
	X = np.vstack([S.flatten(), P.flatten()]).T
	s = X[:,0]
	p = X[:,1]
	y = 1./(1 + 25*(s + p)**2) + 0.5/(1 + 25*(s-0.5)**2) + 0.1/(p+25)

	# Manually check that we sample the same points as in Table 1, CG20x

	# Iteration 0
	I, b, basis, order = paaa(X, y/np.linalg.norm(y), maxiter = 1)
	assert 0. in basis[0]
	assert 0. in basis[1]
	
	I, b, basis, order = paaa(X, y/np.linalg.norm(y), maxiter = 2)
	assert -1. in basis[0]
	assert len(basis[0]) == 2
	assert len(basis[1]) == 1
	
	I, b, basis, order = paaa(X, y/np.linalg.norm(y), maxiter = 3)
	assert np.isclose(np.min(np.abs(0.1 - basis[0])),0)
	assert len(basis[0]) == 3
	assert len(basis[1]) == 1
	
	I, b, basis, order = paaa(X, y/np.linalg.norm(y), maxiter = 4)
	assert np.isclose(np.min(np.abs(1 - basis[1])),0)
	assert len(basis[0]) == 3
	assert len(basis[1]) == 2
	
	I, b, basis, order = paaa(X, y/np.linalg.norm(y), maxiter = 5)
	assert np.isclose(np.min(np.abs(0.6 - basis[1])),0)
	assert len(basis[0]) == 3
	assert len(basis[1]) == 3
	
	I, b, basis, order = paaa(X, y/np.linalg.norm(y), maxiter = 6)
	assert np.isclose(np.min(np.abs(-0.6 - basis[0])),0)
	assert np.isclose(np.min(np.abs(0.1 - basis[1])),0)
	assert len(basis[0]) == 4
	assert len(basis[1]) == 4
	
	I, b, basis, order = paaa(X, y/np.linalg.norm(y), maxiter = 7)
	assert np.isclose(np.min(np.abs(0.6 - basis[0])),0)
	assert np.isclose(np.min(np.abs(0.55 - basis[1])),0)
	assert len(basis[0]) == 5
	assert len(basis[1]) == 5


	# Test the class interface
	aaa = ParametericAAARationalApproximation()
	aaa.fit(X, y)
	err = np.linalg.norm(aaa(X) - y)
	print(err)
	assert err < 1e-10

if __name__ == '__main__':
	test_paaa()
