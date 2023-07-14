import numpy as np
import cvxpy as cp
from vgi.quad_form import QuadForm


def test_eval_U():
    np.random.seed(0)
    U = np.random.randn(5, 5)
    p = np.random.randn(5)
    c = np.random.randn()
    V = QuadForm.random(5)
    V.U = U
    V.p = p
    V.c = c
    x = np.random.randn(5)
    assert V(x) == np.sum(np.square(U @ x)) + 2 * p @ x + c


def test_eval_P():
    np.random.seed(0)
    U = np.random.randn(5, 5)
    P = U.T @ U
    p = np.random.randn(5)
    c = np.random.randn()
    x = np.random.randn(5)
    V = QuadForm.random(5)
    V.U = U
    V.p = p
    V.c = c
    x = np.random.randn(5)
    assert V(x) == np.sum(np.square(U @ x)) + 2 * p @ x + c


def test_standardization_consistency():
    np.random.seed(0)
    V = QuadForm.random(5)
    X_mean = np.random.randn(5)
    X_std = np.random.rand(5)
    V_mean = np.random.randn()
    V_std = np.random.rand()

    x = np.random.randn(5)
    orig = V(x)
    V.apply_standardization(X_mean, X_std, V_mean, V_std)
    V.reverse_standardization(X_mean, X_std, V_mean, V_std)
    assert np.isclose(orig, V(x))


def test_standardization_consistency_grad():
    np.random.seed(0)
    V = QuadForm.random(5)
    X_mean = np.random.randn(5)
    X_std = np.random.rand(5)
    V_mean = np.random.randn(5)
    V_std = np.random.rand(5)
    x = np.random.randn(5)
    orig = V.grad(x)
    V.apply_standardization(X_mean, X_std, V_mean, V_std)
    V.reverse_standardization(X_mean, X_std, V_mean, V_std)
    assert np.allclose(orig, V.grad(x))


def test_minimization():
    np.random.seed(0)
    V = QuadForm.random(5)
    x_min = np.linalg.solve(V.P, -V.p)

    x = cp.Variable(5)
    prob = cp.Problem(cp.Minimize(V.cvxpy_expr(x)))
    prob.solve()
    assert np.allclose(x.value, x_min)
    assert np.isclose(prob.value, V(x_min))


def test_linear_combination():
    np.random.seed(0)
    coefs = [np.random.rand() for _ in range(10)]
    Vs = [QuadForm.random(5) for _ in range(10)]

    Vcomb = QuadForm.linear_combination(coefs, Vs)
    Pcomb = np.sum([a * V.P for a, V in zip(coefs, Vs)], axis=0)
    pcomb = np.sum([a * V.p for a, V in zip(coefs, Vs)], axis=0)
    ccomb = np.sum([a * V.c for a, V in zip(coefs, Vs)], axis=0)

    assert np.allclose(Vcomb.P, Pcomb)
    assert np.allclose(Vcomb.p, pcomb)
    assert np.isclose(Vcomb.c, ccomb)
