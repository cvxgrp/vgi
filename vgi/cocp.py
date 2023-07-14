import numpy as np
import cvxpy as cp
from cvxpygen import cpg
import importlib
import os
from inspect import signature
import warnings

from .parameters import *
from .psd_util import psd_sqrt
from .quad_form import *
from .policy import Policy


class COCP(Policy):

    """Convex optimization control policy (COCP) with a quadratic approximate value function
    Compatible with cvxpygen compilation
    All classes inheriting COCP has the same constructor -- additional arguments can be passed to the constructor,
    and then accessed in the stage_cost function via the self.params dictionary

    stage_cost_params is dictionary of parameters for constructing the stage cost
    CE values are given for lookahead
    values at time t can be updated by passing into __call__ function
    """

    def __init__(
        self,
        n,
        m,
        gamma=1.0,
        lookahead=1,
        V=None,
        A=None,
        B=None,
        c=None,
        samples=1,
        solver=cp.ECOS,
        solver_settings={},
        name="",
        compile=False,
        stage_cost_params={},
        **params
    ):
        # any extra arguments provided
        self.params = params

        # problem dimensions
        self.n = n
        self.m = m

        # check discount and lookahead parameters
        assert 0 < gamma <= 1
        assert lookahead >= 1
        self.gamma = gamma
        self.lookahead = int(np.round(lookahead))

        # validate stage cost function arguments
        parameters = signature(self.stage_cost).parameters.keys()
        assert "x" in parameters
        assert "u" in parameters

        # samples used to estimate EV(Ax+Bu+c)
        self.samples = samples

        # cocp solver and settings
        self.solver = solver
        self.solver_settings = solver_settings.copy()

        # current state parameter
        x = cp.Parameter(n, name="x")

        # stage cost parameters
        self.stage_cost_params = ({}, stage_cost_params)
        for key in stage_cost_params.keys():
            val = stage_cost_params[key]
            self.stage_cost_params[0][key] = cp.Parameter(val.shape, name=key)
            self.stage_cost_params[0][key].value = val

        # affine dynamics parameters
        _A = cp.Parameter((n, n), name="A")
        _B = cp.Parameter((n, m), name="B")
        _c = cp.Parameter(n, name="c")

        # parameters of function EV(Ax+Bu+c)
        H_sqrt = cp.Parameter((n + m, n + m), name="H_sqrt")
        h = cp.Parameter(n + m, name="h")
        self.const = 0

        # state variables
        _x = cp.Variable((lookahead, n), name="_x")
        u = cp.Variable((lookahead, m), name="u")

        # lookahead
        self.x_constaint = _x[0, :] == x
        J, constraints = 0, [self.x_constaint]
        for t in range(lookahead):
            _J, _constraints = self._stage_cost(
                _x[t, :], u[t, :], **self.stage_cost_params[int(t > 0)]
            )
            J += (gamma**t) * _J
            constraints.extend(_constraints)

            # certainty equivalent dynamics for all but terminal state/cost
            if t < lookahead - 1:
                constraints.append(_x[t + 1, :] == _A @ _x[t, :] + _B @ u[t, :] + _c)

        # terminal cost EV(Ax+Bu+c) is quadratic function
        J_terminal = cp.sum_squares(H_sqrt @ cp.hstack((_x[-1, :], u[-1, :])))
        J_terminal += 2 * h @ cp.hstack((_x[-1, :], u[-1, :]))
        J += (gamma**lookahead) * J_terminal

        # set problem
        self.problem = cp.Problem(cp.Minimize(J), constraints)

        # compile or load compiled solution method
        if name != "":
            if compile:
                self.compile(name)
            self.register_solution_method(name)
            self.compiled = True
            self.name = name
        else:
            self.name = ""
            self.compiled = False

        # set any parameters that are given
        self.update_dynamics(A, B, c)
        self.update_value(V)

    def clone(self):
        """create a new instantiation of policy"""
        return type(self)(
            self.n,
            self.m,
            gamma=self.gamma,
            lookahead=self.lookahead,
            V=self.V,
            A=self.A,
            B=self.B,
            c=self.c,
            samples=self.samples,
            solver=self.solver,
            solver_settings=self.solver_settings,
            name=self.name if self.compiled else "",
            stage_cost_params=self.stage_cost_params[1],
            **self.params,
        )

    def register_solution_method(self, policy_name):
        """Register compiled solution method for problem"""
        if policy_name != "":
            pickled_prob_path = CPG_COMPILED_PATH + policy_name
            cpg_solver_module = importlib.import_module(
                CPG_COMPILED_DIR + "." + pickled_prob_path + ".cpg_solver"
            )
            self.problem.register_solve(CPG_METHOD_NAME, cpg_solver_module.cpg_solve)
            self.compiled = True

    def compile(self, policy_name):
        """Create compiled policy core using cvxpygen"""
        # TODO: check this, only ECOS seems to work correctly with cvxpygen
        # compilers for OSQP and SCS don't seem to handle constants in the constraints correctly but don't throw errors
        if not self.solver == cp.ECOS:
            warnings.warn(
                "CVXPY compilation using a solver other than ECOS can currently lead to errors for problems with constraints"
            )

        # path to directory containing compiled policy
        self.policy_name = policy_name
        pickled_prob_path = CPG_COMPILED_PATH + policy_name
        # create directory of compiled policy if it doesn't exist
        if not os.path.isdir(CPG_COMPILED_DIR):
            os.makedirs(CPG_COMPILED_DIR)

        # generate c code using cvxpygen
        cpg.generate_code(
            self.problem,
            code_dir=os.path.join(CPG_COMPILED_DIR, pickled_prob_path),
            prefix=pickled_prob_path,
            solver=self.solver,
        )

    def update_params(self, **param_dict):
        """Valid parameters are H, x, and A, B, c if lookahead > 1"""
        for key in param_dict.keys():
            param = param_dict[key]
            if param is not None and key in self.problem.param_dict.keys():
                self.problem.param_dict[key].value = param
            elif param is not None and key == "const":
                self.const = param

    def update_dynamics(self, A, B, c):
        """Update certainty equivalent dynamics"""
        self.A = A
        self.B = B
        self.c = c
        self.update_params(A=A, B=B, c=c)

    def certainty_equivalent_params(self, V, A, B, c):
        """Calculate EV(Ax+Bu+c) parameters in certainty equivalent case"""
        P, p, pi = V.P, V.p, V.c
        H11 = A.T @ P @ A
        H12 = A.T @ P @ B
        H13 = (A.T @ P @ c + A.T @ p).reshape(-1, 1)
        H22 = B.T @ P @ B
        H23 = (B.T @ P @ c + B.T @ p).reshape(-1, 1)
        H33 = (c.T @ P @ c + 2 * c.T @ p + pi).reshape(1, 1)

        return np.block([[H11, H12, H13], [H12.T, H22, H23], [H13.T, H23.T, H33]])

    def update_value(self, V):
        """Update value function. By default, replaces EV(Ax+Bu+c) with certainty equivalent
        If samples > 1, then EV(Ax+Bu+c) is replaced with a sample average approximation
        """
        self.V = V
        if V is not None:
            M = self.certainty_equivalent_params(V, self.A, self.B, self.c)
            H = M[: self.n + self.m, : self.n + self.m]
            h = M[-1, :-1]
            const = (self.gamma**self.lookahead) * M[-1, -1]
        else:
            H = np.zeros((self.n + self.m, self.n + self.m))
            h = np.zeros(self.n + self.m)
            const = 0

        self.update_params(H_sqrt=psd_sqrt(H), h=h, const=const)

    def __call__(self, x, t, **kwargs):
        """evaluate policy at state x"""
        self.update_params(x=x)

        # set any parameters that are given
        for key in kwargs.keys():
            if key in self.problem.param_dict.keys():
                self.problem.param_dict[key].value = kwargs[key]

        try:
            # use cvxpygen compiled method if available
            if (
                self.compiled
                and CPG_METHOD_NAME in self.problem.REGISTERED_SOLVE_METHODS
            ):
                self.problem.solve(method=CPG_METHOD_NAME, **self.solver_settings)
            else:
                self.problem.solve(solver=self.solver, **self.solver_settings)
        except cp.SolverError:
            warnings.warn("COCP solver failed.")
            return None

        if self.problem.var_dict["u"].value is None:
            warnings.warn("COCP solver failed.")
            return None

        return self.problem.var_dict["u"].value[0, :]

    def steady_state(self):
        """calculate steady state optimal solution"""
        x = cp.Variable(self.n)
        u = cp.Variable(self.m)
        stage_cost, constraints = self._stage_cost(x, u, **self.stage_cost_params[1])
        constraints += [x == self.A @ x + self.B @ u + self.c]
        prob = cp.Problem(cp.Minimize(stage_cost), constraints)
        prob.solve()
        return x.value, u.value

    @property
    def value(self):
        """Get value for fitting value function"""
        return self.problem.value + self.const

    @property
    def value_gradient(self):
        """Get value gradient for fitting value function"""
        return -self.x_constaint.dual_value

    def _stage_cost(self, x, u, **kwargs):
        """wrapped version of stage cost which returns tuple (cost, constraint_list)"""
        x = self.stage_cost(x, u, **kwargs)
        return x if type(x) == tuple else (x, [])

    def stage_cost(self, x, u, **kwargs):
        """Returns stage cost, and optionally a list of constraints. Must be DCP and DPP"""
        raise NotImplementedError
