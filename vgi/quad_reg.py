import numpy as np
import cvxpy as cp

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval, StrOptions
from numbers import Real
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV

from scipy.special import huber

from .psd_util import *
from .quad_form import QuadForm


class QuadReg(BaseEstimator, RegressorMixin):
    """Fit a convex quadratic function V(x) = x^T Px + 2p^Tx + c to data (X, y)
    where X is data matrix with shape (N, n), y is a vector of length N
    """

    _parameter_constraints: dict = {
        "l2_penalty": [Interval(Real, 0.0, None, closed="left")],
        "l1_penalty": [Interval(Real, 0.0, None, closed="left")],
        "diagonal": ["boolean"],
        "V_lb": [None, QuadForm],
        "loss": [StrOptions({"huber", "square"})],
        "symmetric": ["boolean"],
        "standardize_data": ["boolean"],
        "solver": [None, str],
        "solver_settings": [dict],
    }

    def __init__(
        self,
        *,
        l2_penalty=0,
        l1_penalty=0,
        diagonal=False,
        V_lb=None,
        loss="huber",
        symmetric=False,
        standardize_data=False,
        solver=None,
        solver_settings={},
    ):
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.diagonal = diagonal
        self.V_lb = V_lb
        self.loss = loss
        self.symmetric = symmetric
        self.standardize_data = standardize_data
        self.solver = solver
        self.solver_settings = solver_settings

    def fit(self, X, y):
        """Fit a convex quadratic function from imput data"""
        self._validate_params()
        self.X_, self.y_ = self._validate_data(
            X,
            y,
            accept_sparse=("csr", "csc"),
            multi_output=np.ndim(y) > 1,
            y_numeric=True,
        )

        # option to recenter and rescale data
        if self.standardize_data:
            self.X_, self.y_ = self._standardize_data(self.X_, self.y_)
            self.V_lb_ = self._standardize_lower_bound(self.V_lb)
        else:
            self.V_lb_ = None if self.V_lb is None else self.V_lb.clone()

        # solve fitting problem
        self._create_quadratic_params()

        # compute fitting loss
        J = self.fitting_objective()

        # add regularization
        if self.l2_penalty > 0:
            J += self.l2_penalty * cp.sum_squares(
                cp.hstack((cp.vec(self.P), 2 * self.p))
            )
        if self.l1_penalty > 0:
            J += self.l1_penalty * cp.norm(cp.hstack((cp.vec(self.P), 2 * self.p)), 1)

        # add lower bound constraint
        if self.V_lb_ is not None:
            dP = self.P - self.V_lb_.P
            dp = cp.reshape(self.p - self.V_lb_.p, (self.p.shape[0], 1))
            dpi = cp.Variable((1, 1))
            self.constraints.append(cp.bmat([[dP, dp], [dp.T, dpi]]) >> 0)

        # solve fitting problem using cvxpy
        problem = cp.Problem(cp.Minimize(J), self.constraints)
        problem.solve(solver=self.solver, **self.solver_settings)

        # extract solution
        P_ = self.P.value.diagonal() if self.diagonal else self.P.value
        p_ = self.p.value
        c_ = self.c.value
        self.V_ = QuadForm(P=P_, p=p_, c=c_)

        # reverse data standardization
        if self.standardize_data:
            self.V_.reverse_standardization(
                self.X_mean_, self.X_std_, self.y_mean_, self.y_std_
            )

        # expose fitted parameters
        self.P_ = self.V_.P
        self.p_ = self.V_.p
        self.c_ = self.V_.c

        return self

    def fitting_objective(self):
        """Solve fitting problem"""
        loss_fn = cp.huber if self.loss == "huber" else cp.square
        y_hat = (
            cp.sum(cp.multiply(self.X_, self.X_ @ self.P), axis=1)
            + 2 * self.X_ @ self.p
            + self.c
        )
        return cp.sum(loss_fn(y_hat - self.y_))

    def predict(self, X):
        """Evaluate fitted convex quadratic"""
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=("csr", "csc"), reset=False)
        return self.V_(X)

    def score(self, X, y):
        """Score function for hyperparameter tuning"""
        loss_fn = lambda a: 2 * huber(1, a) if self.loss == "huber" else np.square
        y_hat = self.predict(X)
        return -np.mean(loss_fn(y_hat - y))

    def _standardize_data(self, X, y):
        """Preprocess data to have zero mean and unit variance"""
        self.y_mean_ = np.mean(y, axis=0)
        self.y_std_ = np.std(y, axis=0)
        self.X_mean_ = np.mean(X, axis=0)
        if y.ndim == 1:
            # scale X if y is a scalar
            self.X_std_ = np.std(X, axis=0)
        else:
            # scale X by 1/same factors as y if y is a scalar
            self.X_std_ = 1 / self.y_std_

        X = (X - self.X_mean_) / self.X_std_
        y = (y - self.y_mean_) / self.y_std_
        return X, y

    def _standardize_lower_bound(self, V_lb):
        """Apply linear data transformation to parameters of lower bound"""
        if V_lb is None:
            return None
        V_lb_ = V_lb.clone()
        V_lb_.apply_standardization(
            self.X_mean_, self.X_std_, self.y_mean_, self.y_std_
        )
        return V_lb_

    def _create_quadratic_params(self):
        """Create cvxpy variables for fitting problem"""
        n = self.X_.shape[1]
        self.constraints = []

        # diagonal or general PSD matrix
        if self.diagonal:
            self.P = cp.Variable((n, n), diag=True, name="P")
            self.constraints.append(cp.diag(self.P) >= 0)
        else:
            self.P = cp.Variable((n, n), PSD=True, name="P")

        # if symmetric function, set linear term p to zero parameter
        if self.symmetric:
            self.p = cp.Parameter(n, name="p")
            self.p.value = np.zeros(n)
        else:
            self.p = cp.Variable(n, name="p")

        # offset term
        self.c = cp.Variable(name="c")


class QuadGradReg(QuadReg):
    """Fit a convex quadratic function V(x) = x^T Px + 2p^Tx + c to data (X, y)
    where X is data array with shape (N, n), y is an array of gradients (N, n)
    """

    _parameter_constraints: QuadReg._parameter_constraints

    def fitting_objective(self):
        """Solve fitting problem"""
        self.constraints.append(self.c == 0)
        loss_fn = (
            lambda a: cp.huber(cp.norm(a, axis=1))
            if self.loss == "huber"
            else cp.square
        )
        y_hat = 2 * self.X_ @ self.P + 2 * self.p[None, :]
        return cp.sum(loss_fn(y_hat - self.y_))

    def score(self, X, y):
        """Score function for hyperparameter tuning"""
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=("csr", "csc"), reset=False)
        loss_fn = (
            lambda a: 2 * huber(1, np.linalg.norm(a, axis=1))
            if self.loss == "huber"
            else np.square
        )
        return -np.mean(loss_fn(self.V_.grad(X) - y))
