import numpy as np
import cvxpy as cp
from vgi.lower_bound import lqr_ce_value, stoch_lqr_value

"""Test if a value function satisfies Bellman equation for an LQR problem"""


def test_lower_bound():
    def calc_lower_bound(
        V,
        A,
        B,
        Q,
        R,
        Cov,
        gamma=1.0,
        S=None,
        q=None,
        r=None,
        s=0.0,
        w_mean=None,
        tests=100,
    ):
        n, m = B.shape
        assert A.shape == (n, n)
        assert Q.shape == (n, n)
        assert R.shape == (m, m)
        if S is not None:
            assert S.shape == (n, m)
        if q is not None:
            assert len(q == n)
        if r is not None:
            assert len(r == m)
        if w_mean is not None:
            assert len(w_mean == n)

        S = np.zeros((n, m)) if S is None else S
        q = np.reshape(np.zeros(n) if q is None else q, (n, 1))
        r = np.reshape(np.zeros(m) if r is None else r, (m, 1))

        x = cp.Parameter(n, name="x")
        u = cp.Variable(m, name="u")
        xu = cp.hstack((x, u))
        M = np.bmat(
            [
                [Q, S],
                [S.T, R],
            ]
        )
        quad_term = cp.quad_form(xu, M)
        lin_term = q.T @ x + r.T @ u
        const_term = s + gamma * np.trace(V.P @ Cov)
        TVxu = (
            quad_term
            + 2 * lin_term
            + const_term
            + gamma * V.cvxpy_expr(A @ x + B @ u + w_mean)
        )
        prob = cp.Problem(cp.Minimize(TVxu))
        success_ratio = 0.0
        for _ in range(tests):
            _x = np.random.randn(n)
            V_x = V(_x)
            x.value = _x
            prob.solve()
            success_ratio += np.isclose(prob.value, V_x)
        return success_ratio / tests

    """Test lqr_ce_value on a random LQR problem instantiation"""

    def run_lb_test(n, m, gamma=0.99, tests=100, stochastic=False):
        A = np.random.randn(n, n)
        B = np.random.randn(n, m)
        Q = np.diag(np.random.rand(n)) + 1
        R = np.diag(np.random.rand(m)) + 1
        Cov = 0.25 * np.eye(n)
        S = np.random.rand(n, m)
        while np.any(np.linalg.eigvals(np.bmat([[Q, S], [S.T, R]])) < 0):
            S *= 0.5
        q = np.random.randn(n)
        r = np.random.randn(m)
        s = 1.0
        w_mean = 4 * np.ones(n)

        if stochastic:
            _A = lambda: A
            _B = lambda: B
            _c = lambda: np.random.multivariate_normal(w_mean, Cov)
            V_lb = stoch_lqr_value(_A, _B, _c, Q, R, gamma=gamma, S=S, q=q, r=r, s=s)
        else:
            V_lb = lqr_ce_value(
                A, B, Q, R, Cov, gamma=gamma, S=S, q=q, r=r, s=s, w_mean=w_mean
            )
        return calc_lower_bound(
            V_lb,
            A,
            B,
            Q,
            R,
            Cov,
            gamma=gamma,
            S=S,
            q=q,
            r=r,
            s=s,
            w_mean=w_mean,
            tests=tests,
        )

    assert run_lb_test(5, 3) == 1
