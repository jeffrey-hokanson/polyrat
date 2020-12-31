import numpy as np
import polyrat
from polyrat.aaa import _build_cauchy
from polyrat.vecfit_rank import (_fit_f, _fit_g, _fit_h, _fit_fh, _fit_gh)
import cvxpy as cp
import pytest


def generate_test_data(M = 100, r = 10, p = 5, m = 3):
	z = 1j*np.linspace(-1, 1, M)
	lam = 1j*np.linspace(-1, 1,r) - 0.1
	
	As = [np.random.randn(p,1) @ np.random.randn(1,m) for lam_ in lam] 
	#As = [np.eye(p*m)[k % (m*p)].reshape(p,m) for k, lam_ in enumerate(lam)] 
	Hz = np.zeros((M, p, m), dtype = np.complex)
	for lam_, A in zip(lam, As):
		Hz += np.einsum('i,jk->ijk',1/(z - lam_), A)

	return z.reshape(-1,1), Hz, lam


def fit_f(Y, C, g):
	f = cp.Variable((C.shape[1], Y.shape[1]), complex = True)

	obj = 0
	Hr = [0 for i in range(C.shape[0])]
	for i in range(C.shape[0]):
		for k in range(C.shape[1]):
			Hr[i] += (f[k:k+1].T @ g[k:k+1].conj())*C[i,k]
		obj += cp.sum_squares(Y[i] - Hr[i])

	prob = cp.Problem(cp.Minimize(obj))
	prob.solve(verbose = True, solver = 'OSQP',eps_abs = 1e-8, eps_rel = 1e-8)
	return f.value

def fit_g(Y, C, f):
	g = cp.Variable((C.shape[1], Y.shape[2]), complex = True)

	obj = 0
	Hr = [0 for i in range(C.shape[0])]
	for i in range(C.shape[0]):
		for k in range(C.shape[1]):
			Hr[i] += (f[k:k+1].T @ cp.conj(g[k:k+1]) )*C[i,k]
		obj += cp.sum_squares(Y[i] - Hr[i])

	prob = cp.Problem(cp.Minimize(obj))
	prob.solve(verbose = True, solver = 'OSQP',eps_abs = 1e-8, eps_rel = 1e-8)
	return g.value
	

@pytest.mark.parametrize("r", [4])
@pytest.mark.parametrize("p", [1,2])
@pytest.mark.parametrize("m", [1,2])
def test_fit_f(r, p, m):
	np.random.seed(0)
	X, Y, lam = generate_test_data(M = 20, r = r, p = p, m = m)
	poles = lam - 0.1
	C = _build_cauchy(X, poles)

	g = np.random.randn(C.shape[1], Y.shape[2])
	
	f_np = _fit_f(Y, C, g)

	f_true = fit_f(Y, C, g)
	err = np.max(np.abs(f_true - f_np))
	print(err)
	print("my implementation")
	print(f_np)
	print("cvxpy")
	print(f_true)
	assert err < 1e-5

def test_fit_g():
	np.random.seed(0)
	X, Y, lam = generate_test_data()
	poles = lam - 0.1
	C = _build_cauchy(X, poles)

	f = np.random.randn(C.shape[1], Y.shape[1])

	g_np = _fit_g(Y, C, f)

	g_true = fit_g(Y, C, f)
	err = np.max(np.abs(g_true - g_np))
	print("true g")
	print(g_true)
	print("estimated g")
	print(g_np)
	print(err)
	assert err < 1e-5


def fit_h(Y, C, f, g):
	M = Y.shape[0]
	r = C.shape[1]
	h = cp.Variable(r, complex = True)

	obj = 0
	Y = np.copy(Y)
	Ymis = np.copy(Y)
	for i in range(M):
		for k in range(r):
			Ymis[i] -= (f[k:k+1].T @ g[k:k+1].conj())* C[i,k]
		
		obj+= cp.sum_squares(Ymis[i] + Y[i] * (C[i,:] @ h))
		
	prob = cp.Problem(cp.Minimize(obj))
	prob.solve(verbose = True, solver = 'OSQP',eps_abs = 1e-8, eps_rel = 1e-8)
	return h.value


def test_fit_h():
	np.random.seed(0)
	X, Y, lam = generate_test_data()
	poles = lam - 0.1
	C = _build_cauchy(X, poles)

	f = np.random.randn(C.shape[1], Y.shape[1])
	g = np.random.randn(C.shape[1], Y.shape[2])

	h_est = _fit_h(Y, C, f, g)

	h_true = fit_h(Y, C, f, g)
	err = np.max(np.abs(h_true - h_est))
	print(err)
	print("cvxpy")
	print(h_true)
	print("mine")
	print(h_est)
	assert err < 1e-5
	


def fit_fh(Y, C, g):
	f = cp.Variable((C.shape[1], Y.shape[1]), complex = True)
	h = cp.Variable(C.shape[1], complex = True)
	obj = 0
	Hr = [0 for i in range(C.shape[0])]
	for i in range(C.shape[0]):
		for k in range(C.shape[1]):
			Hr[i] += (f[k:k+1].T @ g[k:k+1].conj())*C[i,k]
		obj += cp.sum_squares(Y[i]*(1 + C[i,:]@ h) - Hr[i])

	prob = cp.Problem(cp.Minimize(obj))
	prob.solve(verbose = True, solver = 'OSQP',eps_abs = 1e-8, eps_rel = 1e-8)
	return f.value, h.value



def test_fit_fh():
	
	np.random.seed(0)
	X, Y, lam = generate_test_data()
	poles = lam - 0.1
	C = _build_cauchy(X, poles)

	g = np.random.randn(C.shape[1], Y.shape[2])

	f, h = _fit_fh(Y, C, g)

	f_true, h_true = fit_fh(Y, C, g)

	err_f = np.max(np.abs(f - f_true))
	err_h = np.max(np.abs(h - h_true))

	print("---true---")
	print("f")
	print(f_true)
	print("h")
	print(h_true)
	print('---solver---')
	print("f")
	print(f)
	print("h")
	print(h)
	
	print("\n")
	print(f"Error: f:{err_f}, h:{err_h}")
	assert err_f < 1e-5
	assert err_h < 1e-5


def fit_gh(Y, C, f):
	g = cp.Variable((C.shape[1], Y.shape[2]), complex = True)
	h = cp.Variable(C.shape[1], complex = True)

	obj = 0
	Hr = [0 for i in range(C.shape[0])]
	for i in range(C.shape[0]):
		for k in range(C.shape[1]):
			Hr[i] += (f[k:k+1].T @ cp.conj(g[k:k+1]) )*C[i,k]
		obj += cp.sum_squares(Y[i]*(1 + C[i,:] @ h) - Hr[i])

	prob = cp.Problem(cp.Minimize(obj))
	prob.solve(verbose = True, solver = 'OSQP',eps_abs = 1e-8, eps_rel = 1e-8)
	return g.value, h.value

def test_fit_gh():
	np.random.seed(0)
	X, Y, lam = generate_test_data()
	poles = lam - 0.1
	C = _build_cauchy(X, poles)

	f = np.random.randn(C.shape[1], Y.shape[1])

	g, h = _fit_gh(Y, C, f)
	g_true, h_true = fit_gh(Y, C, f)

	err_g = np.max(np.abs(g - g_true))
	err_h = np.max(np.abs(h - h_true))
	
	print("---true---")
	print("g")
	print(g_true)
	print("h")
	print(h_true)
	print('---solver---')
	print("g")
	print(g)
	print("h")
	print(h)
	
	print("\n")
	print(f"Error: g:{err_g}, h:{err_h}")
	assert err_g < 1e-5
	assert err_h < 1e-5


def test_vecfit_rank():
	np.random.seed(0)
	X, Y, lam = generate_test_data(M = 500, r = 5)
	poles = lam - 10e-1
	poles = lam + 1e-3*(np.random.randn(*poles.shape) + 1j*np.random.randn(*poles.shape))

	r = len(poles)
	polyrat.vecfit_rank(X, Y, r - 1, r, poles0 = poles)


#	poles0 =  lam - 0.1
#	polyrat.vecfit_rank(z.reshape(-1,1), Hz, 9,10, poles0 = poles0)
#
#
#	X = np.arange(12).reshape(3,4)
#	print(X)
#	print(X.flatten())
#	print(X.flatten('F'))

if __name__ == '__main__':
	test_vecfit_rank()
#	test_fit_f()
#	test_fit_g()
#	test_fit_h()
#	test_fit_fh()
#	test_fit_gh()
