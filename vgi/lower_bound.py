import numpy as np
import scipy as sp
import cvxpy as cp

from .quad_form import QuadForm


"""Calculate exact value function for an LQR problem"""


def lqr_ce_value(
    A, B, Q, R, Cov, gamma=1.0, S=None, q=None, r=None, s=0.0, w_mean=None
):
    """Compute true value function for a deterministic/certainty equivalent LQR problem"""
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

    P = sp.linalg.solve_discrete_are(np.sqrt(gamma) * A, np.sqrt(gamma) * B, Q, R, s=S)
    S = np.zeros((n, m)) if S is None else S
    q = np.zeros(n) if q is None else q
    r = np.zeros(m) if r is None else r
    w_mean = np.zeros(n) if w_mean is None else w_mean

    Pxu = S + gamma * A.T @ P @ B
    Puu_inv = np.linalg.pinv(R + gamma * B.T @ P @ B)
    Ap = np.eye(n) - gamma * A.T + gamma * Pxu @ Puu_inv @ B.T
    bp = q + gamma * (A.T @ P @ w_mean) - Pxu @ Puu_inv @ (r + gamma * B.T @ P @ w_mean)
    p = np.linalg.solve(Ap, bp)

    ru = r + gamma * (B.T @ p + B.T @ P @ w_mean)
    if gamma < 1:
        pi = (
            s
            + gamma * (np.trace(P @ Cov) + w_mean.T @ P @ w_mean + 2 * p @ w_mean)
            - ru.T @ Puu_inv @ ru
        ) / (1 - gamma)
    else:
        pi = 0

    return QuadForm(P=P, p=p, c=pi)


def lqr_cost_to_go(V_lb, x0_mean, cov_x0):
    """Compute cost of an LQR problem from value function"""
    return V_lb.at(x0_mean) + np.trace(V_lb.P @ cov_x0)


def lqr_average_cost(
    V_lb, A, B, Q, R, Cov, S=None, q=None, r=None, s=0.0, w_mean=None, x_ref=None
):
    """Compute estimated average cost via TV(x^ref)"""
    n, m = B.shape
    S = np.zeros((n, m)) if S is None else S
    q = (np.zeros(n) if q is None else q).reshape(-1, 1)
    r = (np.zeros(m) if r is None else r).reshape(-1, 1)
    w_mean = np.zeros(n) if w_mean is None else w_mean
    x_ref = np.zeros(n) if x_ref is None else x_ref

    u = cp.Variable(m)
    J = x_ref.T @ Q @ x_ref
    J += cp.quad_form(u, R)
    J += 2 * x_ref.T @ S @ u
    J += 2 * q.T @ x_ref
    J += 2 * r.T @ u
    J += V_lb.expr_at(A @ x_ref + B @ u + w_mean)
    J += np.trace(V_lb.P @ Cov)
    prob = cp.Problem(cp.Minimize(J))
    prob.solve()
    return prob.value


def quadratic_box_bound(lower, upper, dim, curvature):
    """Calculate a quadratic lower bound to a box constraint to [lower, upper] on each dimension"""
    assert upper > lower
    c = curvature * lower * upper
    b = -0.5 * curvature * (lower + upper) * np.ones(dim)
    return QuadForm(P=curvature * np.eye(dim), p=b, c=c)


def linear_quadratic_vi(A, B, Q, w, gamma=0.99, thresh=1e-6, maxiter=5000, P=None):
    n, m = B.shape
    assert A.shape == (n, n)

    # do not fit constant term
    if P is None:
        P = np.zeros((n + 1, n + 1))
    P[-1, -1] = 0

    prev_infnorm = np.inf
    for i in range(1, maxiter + 1):
        _P = P[:n, :n]
        _p = P[-1, :n]
        _pi = P[-1, -1]

        MM = np.block(
            [
                [A.T @ _P @ A, A.T @ _P @ B, (A.T @ _p + A.T @ _P @ w).reshape(-1, 1)],
                [B.T @ _P @ A, B.T @ _P @ B, (B.T @ _p + B.T @ _P @ w).reshape(-1, 1)],
                [
                    (_p.T @ A + w.T @ _P @ A).reshape(1, -1),
                    (_p.T @ B + w.T @ _P @ B).reshape(1, -1),
                    (2 * _p.T @ w + w.T @ _P @ w + _pi) * np.ones((1, 1)),
                ],
            ]
        )
        # ignore constant term (2 * _p.T @ w + _l + w.T @ _P @ w + _pi)*np.ones((1,1))
        M = Q + gamma * MM

        Mxx = M[:n, :n]
        Mxu = M[:n, n : n + m]
        Muu = M[n : n + m, n : n + m]
        qx = M[-1, :n]
        qu = M[-1, n : n + m]
        Mr = M[-1, -1]

        Minv = np.linalg.inv(Muu)
        P_next = np.block(
            [
                [Mxx - Mxu @ Minv @ Mxu.T, (qx - Mxu @ Minv @ qu).reshape(-1, 1)],
                [
                    (qx.T - qu.T @ Minv @ Mxu.T).reshape(1, -1),
                    (Mr - qu.T @ Minv @ qu) * np.ones((1, 1)),
                ],
            ]
        )
        scaled_infnorm = np.max(np.abs(P_next - P)) / ((n + 1) ** 2)

        # stop if numerically unstable
        if scaled_infnorm > prev_infnorm and i > 5:
            return P
        if scaled_infnorm < thresh:
            return P_next
        P = P_next
        prev_infnorm = scaled_infnorm
    return P


def stoch_lqr_value(
    _A,
    _B,
    _c,
    Q,
    R,
    gamma=1.0,
    S=None,
    q=None,
    r=None,
    s=0.0,
    max_iters=1000,
    gap=1e-3,
    samples=1,
):
    """Perform value iteration for a stochastic LQR problem, estimating expected
    values via sample average approximation

    In each iteration, V is updated to
    \min_{u} (x' Q x + u' R u + 2 x' S u + 2 x' q + 2 u' r + s + gamma E V(x + Aw + Bu))
    """
    n = Q.shape[0]
    m = R.shape[0]

    S = np.zeros((n, m)) if S is None else S
    q = np.zeros(n) if q is None else q
    r = np.zeros(m) if r is None else r

    M = np.block(
        [
            [Q, S, q.reshape(-1, 1)],
            [S.T, R, r.reshape(-1, 1)],
            [q.reshape(1, -1), r.reshape(1, -1), s],
        ]
    )

    P = np.zeros((n, n))
    p = np.zeros(n)
    pi = 0

    prev_diff = np.inf
    for i in range(max_iters):
        # calculate parameters of EV(Ax + Bu + c)
        EV = np.zeros((n + m + 1, n + m + 1))
        for _ in range(samples):
            A = _A()
            B = _B()
            c = _c()
            EV11 = A.T @ P @ A
            EV12 = A.T @ P @ B
            EV13 = (A.T @ P @ c + A.T @ p).reshape(-1, 1)
            EV22 = B.T @ P @ B
            EV23 = (B.T @ P @ c + B.T @ p).reshape(-1, 1)
            EV33 = (c.T @ P @ c + 2 * c.T @ p + pi).reshape(1, 1) * 0
            EV += (
                np.block(
                    [
                        [EV11, EV12, EV13],
                        [EV12.T, EV22, EV23],
                        [EV13.T, EV23.T, EV33],
                    ]
                )
                / samples
            )

        # partial minimization of M + gamma * EV
        G = M + gamma * EV
        Gxx = G[:n, :n]
        Gxu = G[:n, n : n + m]
        Guu = G[n : n + m, n : n + m]
        qx = G[:n, n + m]
        qu = G[n : n + m, n + m]
        Gr = G[n + m, n + m]
        Guuinv = np.linalg.inv(Guu)

        P_ = Gxx - Gxu @ Guuinv @ Gxu.T
        p_ = qx - Gxu @ Guuinv @ qu
        pi_ = (Gr - qu.T @ Guuinv @ qu) * 0

        # check convergence
        diff = max(np.abs(P - P_).max(), np.abs(p - p_).max())
        print(diff)
        # if (diff > prev_diff and i > 10) or diff < gap:
        if diff < gap:
            break

        prev_diff = diff

        P = P_.copy()
        p = p_.copy()
        pi = pi_

    return QuadForm(P=P, p=p, c=pi)
