import numpy as np
import cvxpy as cp
from numbers import Real
from .psd_util import *


class QuadForm:
    """Convex quadratic function of the form V(x) = x.T @ U.T @ U @ x + 2 * p @ x + c"""

    def __init__(self, U=None, p=None, c=0, *, P=None):
        if P is not None:
            self.P = P
        else:
            self.U = U.copy()
        self.n = self.U.shape[1]
        self.p = np.zeros(self.n) if p is None else p.copy().flatten()
        assert len(p) == self.n
        self.c = 0.0 if c is None else c

    def __call__(self, x):
        """evaluate at a point x or an array of N points X
        input: x, shape (n,) or (N, n)"""
        if x.ndim > 1:
            quad = np.multiply(x, x @ self.P).sum(axis=1)
        else:
            quad = np.square(np.linalg.norm(self.U @ x))
        return quad + 2 * x @ self.p + self.c

    def grad(self, x):
        """evaluate gradient at a point x or an array of N points X
        input: numpy array x, shape (n,) or (N, n)"""
        if x.ndim > 1:
            return 2 * x @ self.P + 2 * self.p[None, :]

        return 2 * self.P @ x + 2 * self.p

    def cvxpy_expr(self, x):
        """cvxpy expression of value function at a variable point x
        input: cvxpy variable/expression x, shape (n,)"""
        return cp.sum_squares(self.U @ x) + 2 * self.p @ x + self.c

    @property
    def U(self):
        """get factor"""
        return self.U_

    @property
    def P(self):
        """get PSD matrix"""
        return self.P_

    @U.setter
    def U(self, U_new):
        """set factor"""
        self.U_ = U_new
        self.P_ = U_new.T @ U_new

    @P.setter
    def P(self, P_new):
        """set PSD matrix"""
        self.U = psd_sqrt(project_psd(P_new))

    @property
    def params(self):
        """Get vector of parameters"""
        return np.hstack((self.P.reshape(-1), 2.0 * self.p, self.c))

    @params.setter
    def params(self, theta):
        """Set parameters from a vector of parameters"""
        self.P = theta[: self.n**2].reshape((self.n, self.n))
        self.p = 0.5 * theta[self.n**2 : self.n**2 + self.n]
        self.c = theta[-1]

    def apply_standardization(self, X_mean, X_std, V_mean, V_std):
        """Transform function in state space to function in standardized state space"""
        if isinstance(V_mean, Real) and isinstance(V_std, Real):
            # scalar outputs -- standardization of value
            X_std_diag = np.diag(X_std)

            P_prime = X_std_diag @ self.P @ X_std_diag / V_std
            p_prime = X_std_diag @ (self.p + self.P @ X_mean) / V_std
            c_prime = (
                self.c - V_mean + 2 * self.p @ X_mean + X_mean.T @ self.P @ X_mean
            ) / V_std

            self.P = P_prime
            self.p = p_prime
            self.c = c_prime
        else:
            # vector outputs -- standardization of value gradient
            Cinv = np.diag(1 / V_std)
            self.p = Cinv @ (self.P @ X_mean + self.p - 0.5 * V_mean)
            self.P = Cinv @ self.P @ Cinv

    def reverse_standardization(self, X_mean, X_std, V_mean, V_std):
        """Transform function in standardized state space to function in state space"""
        if isinstance(V_mean, Real) and isinstance(V_std, Real):
            # scalar outputs -- standardization of value

            X_std_inv = np.diag(1.0 / X_std)
            P_prime = V_std * X_std_inv @ self.P @ X_std_inv
            p_prime = V_std * X_std_inv @ self.p - P_prime @ X_mean
            c_prime = (
                V_std * self.c
                + V_mean
                - X_mean.T @ P_prime @ X_mean
                - 2 * p_prime @ X_mean
            )

            self.P = P_prime
            self.p = p_prime
            self.c = c_prime
        else:
            # vector outputs -- standardization of value gradient
            C = np.diag(V_std)
            self.P = C @ self.P @ C
            self.p = C @ self.p - self.P @ X_mean + 0.5 * V_mean

    def clone(self):
        """Return a copy"""
        return QuadForm(self.U, self.p, self.c)

    @staticmethod
    def random(n):
        """Return a random quadratic form"""
        U = np.random.randn(n, n)
        return QuadForm(U, np.random.randn(n), np.random.randn())

    @staticmethod
    def zero(n):
        """Return a zero quadratic form"""
        return QuadForm(np.zeros((n, n)), np.zeros(n), 0)

    @staticmethod
    def eye(n):
        """Return an identity quadratic form"""
        return QuadForm(np.eye(n), np.zeros(n), 0)

    @staticmethod
    def linear_combination(coefs, Vs):
        """Linear combination of convex quadratic functions"""
        Vcomb = Vs[0].clone()
        Vcomb.params = np.sum([a * V.params for a, V in zip(coefs, Vs)], axis=0)
        return Vcomb
