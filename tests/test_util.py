import numpy as np
from polyrat import *
from polyrat.util import *

import pytest

@pytest.mark.parametrize("output_dim",[ (), (1,), (2,), (2,1), (2,2,1)])
@pytest.mark.parametrize("complex_", [True, False])
def test_linearized(output_dim, complex_):
	#complex_ = True
	M = 50
	#output_dim = ()
	if complex_:
		X = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
	else:
		X = np.random.randn(M, 1)

	P = LegendrePolynomialBasis(X, 2).vandermonde_X
	Q = LegendrePolynomialBasis(X, 2).vandermonde_X

	Y = np.random.randn(M, *output_dim)
	A = linearized_ratfit_operator_dense(P, Q, Y)

	As = LinearizedRatfitOperator(P, Q, Y)

	assert np.all(A.shape == As.shape), "Dimensions do not match"

	X = np.random.randn(A.shape[1], 3)
	diff = np.linalg.norm(A @ X - As @ X, 'fro')
	print("forward error", diff)
	assert np.all(np.isclose(A @ X, As @ X))

	# Now check the adjoints
	Z = np.random.randn(A.shape[0], 3)

	diff = np.linalg.norm( As.adjoint() @ Z - A.conj().T @ Z, 'fro')
	print("adjoint error", diff)
	assert np.all(np.isclose(As.adjoint() @ Z, A.conj().T @ Z))



# NOTE: Sparse test fails

@pytest.mark.parametrize("output_dim",[ (), (1,), (2,), (2,1), (2,2,1)])
@pytest.mark.parametrize("complex_", [True, False])
@pytest.mark.parametrize("method", ['dense', 'varpro'])
@pytest.mark.parametrize("Basis", [LegendrePolynomialBasis, ArnoldiPolynomialBasis])
def test_minimize_2norm_exact(output_dim, complex_, method, Basis):
	np.random.seed(0)
	#complex_ = True
	M = 100
	#output_dim = ()
	if complex_:
		X = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
	else:
		X = np.random.randn(M, 1)

	num_basis = Basis(X, 4)
	P = num_basis.vandermonde_X
	denom_basis = Basis(X, 5)
	Q = denom_basis.vandermonde_X

	a0 = np.random.randn(P.shape[1], *output_dim)
	b0 = np.random.randn(Q.shape[1])
	
	num0 = Polynomial(num_basis, a0)
	denom0 = Polynomial(denom_basis, b0)
	rat0 = RationalRatio(num0, denom0)
	
	# Check if we get the exact fit
	Y = rat0(X) 
	#Y += 1e-3*np.random.randn(*Y.shape)

	if method == 'dense':
		a, b, cond = minimize_2norm_dense(P, Q, Y)
	elif method == 'sparse':
		a, b, cond = minimize_2norm_sparse(P, Q, Y)
	elif method == 'varpro':
		if Basis == ArnoldiPolynomialBasis:	
			a, b, cond = minimize_2norm_varpro(P, Q, Y, P_orth = True)
		else:
			a, b, cond = minimize_2norm_varpro(P, Q, Y, P_orth = False)

	else:
		raise NotImplementedError
	
	print("a", a/b[0])
	print("a0", a0/b0[0])

	print("b", b/b[0])
	print("b0", b0/b0[0])
	assert np.all(np.isclose(a/b[0], a0/b0[0])), "a does not match"
	assert np.all(np.isclose(b/b[0], b0/b0[0])), "b does not match"

	num = Polynomial(num_basis, a)
	denom = Polynomial(denom_basis, b)
	rat = RationalRatio(num, denom)

	# We should reproduce exactly	
	err_abs= np.max(np.abs(rat(X) - Y))
	print("Fit error (abs norm)", err_abs)
	err2 = np.linalg.norm( (rat(X) - Y).flatten(), 2)
	print("fit error (2 norm)", err2)
	assert err_abs < 1e-10

if __name__ == '__main__':
	test_minimize_2norm_exact((3,2), False, 'dense', LegendrePolynomialBasis)
	pass
#	test_linearized()	
