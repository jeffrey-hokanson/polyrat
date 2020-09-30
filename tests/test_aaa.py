import numpy 
from polyrat.aaa import *
import pytest

def eval_aaa_slow(xeval, x, y, I, b):
	reval = []
	for i in range(len(xeval)):
		idx = np.isclose(xeval[i], x[I]).flatten()
		if np.any(idx):
			r = (y[I][np.argwhere(idx).flatten()])
			# Make sure dimensions are consistent
			if len(y[0].shape) >0:
				r = r.reshape(y[0].shape)
		else:
			num = 0
			denom = 0
			for j,k in enumerate(np.argwhere(I).flatten()):
				num += y[k] * b[j]/(xeval[i] - x[k])
				denom += b[j]/(xeval[i] - x[k])
			r = num/denom
		reval.append(r)


	return np.array(reval)

def test_eval_aaa():
	
	M = 100
	x = np.linspace(-1,1, M).reshape(-1,1)
	y = np.abs(x).flatten()

	xeval = np.linspace(-1,1, 2*M).reshape(-1,1)

	I = np.zeros(M, dtype = np.bool)
	I[0] = 1
	I[5] = 1
	I[10] = 1
	b = np.arange(np.sum(I))

	reval = eval_aaa(xeval, x, y, I, b)
	reval2 = eval_aaa_slow(xeval, x, y, I, b)
	for r, r2 in zip(reval, reval2):
	# Compute the true value		
		print(f"true", r,  "eval", r2)
		assert np.all(np.isclose(r, r2))

	# Check a matrix version
	y = np.random.randn(M, 2, 3)
	
	reval = eval_aaa(xeval, x, y, I, b)
	reval2 = eval_aaa_slow(xeval, x, y, I, b)
	for r, r2 in zip(reval, reval2):
	# Compute the true value		
		print(f"true", r,  "eval", r2)
		assert np.all(np.isclose(r, r2))


def test_aaa():
	r""" Functionality testing
	"""
	M = int(1e2)
	x = np.linspace(-1,1, M).reshape(-1,1)
	y = np.abs(x).flatten()

	I, b = aaa(x, y, tol = 0)

	# Run to a fixed degree
	I, b = aaa(x, y, degree = 10)

	# Use matrix valued data	
	Y = np.vstack([y for i in range(5)]).T
	I, b = aaa(x, Y, degree = 10)


@pytest.mark.parametrize("M", [1000])
@pytest.mark.parametrize("degree", [5, 10, None])
@pytest.mark.parametrize("tol", [1e-5, 1e-10, None])
@pytest.mark.parametrize("verbose", [True, False])
def test_AAARationalApproximation(M,degree, tol, verbose):
	r""" Integration tests for AAARationalApproximation
	"""
	X = np.random.randn(M,1)
	y = np.abs(X).flatten()
	
	aaa = AAARationalApproximation(degree= degree, tol = tol, verbose = verbose)
	aaa.fit(X, y) 

	res_norm = np.linalg.norm(aaa(X) - y)
	print(res_norm)	

if __name__ == '__main__':
	test_eval_aaa()
#	test_aaa()
