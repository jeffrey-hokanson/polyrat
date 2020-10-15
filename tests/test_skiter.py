import numpy as np
from polyrat import *
from polyrat.skiter import _minimize_inf_norm_real, _minimize_inf_norm_complex

def test_minimize_inf_norm_real():
	np.random.seed(0)
	A = np.random.randn(100, 10)
	x, cond = _minimize_inf_norm_real(A)
	
	assert np.isclose(np.linalg.norm(x,2),1)

	# Compare best solution against random solutions
	for i in range(1000):
		xi = np.random.randn(*x.shape)
		xi /= np.linalg.norm(xi)
		assert np.linalg.norm(A @ x, np.inf) <= np.linalg.norm(A @ xi, np.inf)

def test_minimize_inf_norm_complex():
	np.random.seed(0)
	A = np.random.randn(100, 10) + 1j*np.random.randn(100,10)
	x, cond = _minimize_inf_norm_complex(A, nsamp = 360)
	
	assert np.isclose(np.linalg.norm(x,2),1)

	# Compare best solution against random solutions
	obj = np.linalg.norm(A @ x, np.inf)

	for i in range(1000):
		xi = np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape)
		xi /= np.linalg.norm(xi)

		obj_new = np.linalg.norm(A @ xi, np.inf)
		print(obj_new - obj)
		assert obj <= obj_new


def test_skfit_rebase():
	X = np.random.randn(50,1) #+ 1j*np.random.randn(100,1)
	#y = np.random.randn(100,) + 1j*np.random.randn(100,)
	y = np.abs(X).flatten()

	skfit_rebase(X, y, 4, 4, norm = np.inf)

if __name__ == '__main__':
	#test_minimize_inf_norm_complex()	
	test_skfit_rebase()
