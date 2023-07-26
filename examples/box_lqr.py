import os
import numpy as np
import cvxpy as cp

import torch
from cvxpylayers.torch import CvxpyLayer

from vgi import ControlProblem, COCP, psd_sqrt, QuadForm


class BoxLQRProblem(ControlProblem):
    """Box-constrained LQR problem"""

    def __init__(
        self,
        A,
        B,
        Q,
        R,
        u_max=np.inf,
        x0_var=1.0,
        c_var=1.0,
        state_space_radius=1e4,
        eval_horizon=2 * 10**4,
        eval_trajectories=5,
        processes=1,
    ):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        n, m = B.shape
        self.u_max = u_max
        self.c_var = c_var
        self.x0_var = x0_var
        super().__init__(
            n,
            m,
            1.0,
            state_space_radius,
            eval_horizon,
            eval_trajectories,
            processes,
        )

    def step(self, x, u):
        c = np.sqrt(self.c_var) * np.random.randn(self.n)
        x_next = self.A @ x + self.B @ u + c
        stage_cost = x.T @ self.Q @ x + u.T @ self.R @ u
        return x_next, c, stage_cost

    def sample_initial_condition(self):
        return np.sqrt(self.x0_var) * np.random.randn(self.n)

    def sample_random_control(self, x, t):
        if np.isfinite(self.u_max):
            return self.u_max * (2 * np.random.rand(self.m) - 1)
        return np.random.randn(self.m)

    def create_policy(
        self,
        lookahead=1,
        V=None,
        name="",
        compile=False,
        **kwargs,
    ):
        return BoxLQRPolicy(
            self.n,
            self.m,
            gamma=self.gamma,
            lookahead=lookahead,
            V=V if V is not None else QuadForm.zero(self.n),
            A=self.A,
            B=self.B,
            c=np.zeros(self.n),
            Q=self.Q,
            R=self.R,
            u_max=self.u_max,
            name=name,
            compile=compile,
            **kwargs,
        )

    def J_lb(self):
        """Compute optimized lower bound on optimal cost"""
        W = self.c_var * np.eye(self.n)
        P = cp.Variable((self.n, self.n), PSD=True)
        R = cp.Variable((self.m, self.m), PSD=True)

        if np.isfinite(self.u_max):
            lam = cp.Variable(self.m, nonneg=True)
            objective = cp.trace(P @ W) - (self.u_max**2) * cp.sum(lam)
            constraints = [R - self.R << cp.diag(lam), P >> 0, R >> 0, lam >= 0]
        else:
            objective = cp.trace(P @ W)
            constraints = [P >> 0, R == self.R]

        constraints += [
            cp.bmat(
                [
                    [R + self.B.T @ P @ self.B, self.B.T @ P @ self.A],
                    [self.A.T @ P @ self.B, self.Q + self.A.T @ P @ self.A - P],
                ]
            )
            >> 0
        ]
        Jlb = cp.Problem(cp.Maximize(objective), constraints).solve()
        Vlb = QuadForm(P.value, np.zeros(self.n), 0.0)
        return Jlb, Vlb

    def V_lb(self):
        """Compute optimized lower bound on value function"""
        return self.J_lb()[1]

    @staticmethod
    def create_problem_instance(
        n,
        m,
        u_max=0.4,
        x0_var=0.4,
        c_var=0.4,
        seed=0,
        eval_horizon=2 * 10**4,
        eval_trajectories=5,
        processes=1,
    ):
        """Generate random problem instance"""
        seed = (
            int.from_bytes(os.urandom(4), byteorder="little") if seed is None else seed
        )
        np.random.seed(seed)

        A = 2 * np.random.rand(n, n) - 1
        A /= np.max(np.abs(np.linalg.eigvals(A)))
        B = np.random.rand(n, m) - 1
        Q = np.eye(n)
        R = np.eye(m)
        return BoxLQRProblem(
            A,
            B,
            Q,
            R,
            u_max=u_max,
            c_var=c_var,
            x0_var=x0_var,
            state_space_radius=1e4,
            eval_horizon=eval_horizon,
            eval_trajectories=eval_trajectories,
            processes=processes,
        )


class BoxLQRPolicy(COCP):
    """Quadratic COCP for LQR problem,"""

    def stage_cost(self, x, u):
        if np.isfinite(self.u_max):
            constraints = [u >= -self.u_max, u <= self.u_max]
        else:
            constraints = []
        return (
            cp.sum_squares(psd_sqrt(self.Q) @ x) + cp.sum_squares(psd_sqrt(self.R) @ u),
            constraints,
        )


def box_lqr_cvxpylayer(problem):
    n, m = problem.n, problem.m
    A, B, Q, R = problem.A, problem.B, problem.Q, problem.R

    gamma = problem.gamma
    u_max = problem.u_max

    x = cp.Parameter((n, 1))
    P_sqrt = cp.Parameter((n, n))

    u = cp.Variable((m, 1))
    xnext = cp.Variable((n, 1))

    objective = cp.quad_form(u, R) + gamma * cp.sum_squares(P_sqrt @ xnext)
    constraints = [xnext == A @ x + B @ u, cp.norm(u, "inf") <= u_max]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    policy = CvxpyLayer(prob, [x, P_sqrt], [u])

    return policy


def box_lqr_cocp_grad(
    problem,
    samples_per_iter,
    num_iters,
    learning_rate,
    num_trajectories=1,
    seed=None,
    V0=None,
    Vlb=None,
    policy=None,
    l2_penalty=0.0,
    eval_freq=None,
    restart_simulations=False,
):
    """Differentiate through COCP"""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="little")
    torch.manual_seed(seed)
    np.random.seed(seed)

    eval_freq = num_iters if eval_freq is None else eval_freq
    n = problem.n
    A, B, Q, R = problem.A, problem.B, problem.Q, problem.R
    Qt, Rt, At, Bt = map(torch.from_numpy, [Q, R, A, B])
    torch_policy = box_lqr_cvxpylayer(problem)

    # define loss
    def lossf(time_horizon, batch_size, P_sqrt, x_batch=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if x_batch is None:
            x_batch = np.sqrt(problem.c_var) * torch.randn(batch_size, n, 1).double()
        P_sqrt_batch = P_sqrt.repeat(batch_size, 1, 1)
        Qt_batch = Qt.repeat(batch_size, 1, 1)
        Rt_batch = Rt.repeat(batch_size, 1, 1)
        At_batch = At.repeat(batch_size, 1, 1)
        Bt_batch = Bt.repeat(batch_size, 1, 1)
        loss = 0.0
        for _ in range(time_horizon):
            (u_batch,) = torch_policy(x_batch, P_sqrt_batch)
            state_cost = torch.bmm(
                torch.bmm(Qt_batch, x_batch).transpose(2, 1), x_batch
            )
            control_cost = torch.bmm(
                torch.bmm(Rt_batch, u_batch).transpose(2, 1), u_batch
            )
            loss += (state_cost.squeeze() + control_cost.squeeze()).sum() / (
                time_horizon * batch_size
            )

            c_batch = np.sqrt(problem.c_var) * torch.randn(batch_size, n, 1).double()
            x_batch = (
                torch.bmm(At_batch, x_batch) + torch.bmm(Bt_batch, u_batch) + c_batch
            )

        if l2_penalty > 0.0:
            loss += l2_penalty * (torch.sum(P_sqrt**2))

        return loss, x_batch.detach()

    # initialize parameters
    if V0 is None:
        torch.manual_seed(seed)
        P_sqrt = torch.from_numpy(np.eye(problem.n))
    else:
        P_sqrt = torch.from_numpy(V0.U)
    P_sqrt.requires_grad_(True)

    # run optimization
    opt = torch.optim.Adam(
        [
            P_sqrt,
        ],
        lr=learning_rate,
    )
    costs = []
    value_iterates = []
    x0 = None
    for k in range(num_iters):
        with torch.no_grad():
            U = P_sqrt.detach().numpy()
            V = QuadForm(U, np.zeros(n), 0.0)
            value_iterates.append(V)

            if (k % eval_freq == 0 or k == num_iters - 1) and policy is not None:
                policy.update_value(V)
                expected_cost = problem.cost(
                    policy.clone(), seed=seed + num_iters + k + 1
                )
                test_loss = lossf(
                    samples_per_iter,
                    num_trajectories,
                    P_sqrt.detach(),
                    seed=0,
                )[0].item()

                costs.append(test_loss)
                print(
                    "it: %03d, test loss: %3.3f, policy cost: %3.3f"
                    % (k + 1, test_loss, expected_cost)
                )
            else:
                print("it: %03d" % (k + 1))

        # gradient step
        opt.zero_grad()
        l, x0 = lossf(
            samples_per_iter,
            num_trajectories,
            P_sqrt,
            x_batch=None if restart_simulations else x0,
            seed=seed + k + 1,
        )
        l.backward()
        opt.step()

        # enforce lower bound
        if Vlb is not None:
            P_sqrt_npy = P_sqrt.detach().numpy()
            P_curr = P_sqrt_npy.T @ P_sqrt_npy
            _P = cp.Variable((problem.n, problem.n), PSD=True)
            proj = cp.Problem(
                cp.Minimize(cp.sum_squares(_P - P_curr)),
                [_P >> P_curr],
            )
            proj.solve()

            P_sqrt.data.fill_(0)
            P_sqrt.data += torch.from_numpy(psd_sqrt(_P.value))

    return {"costs": costs, "iterates": value_iterates}
