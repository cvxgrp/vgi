import numpy as np
import cvxpy as cp
import os
import torch
from cvxpylayers.torch import CvxpyLayer


from vgi import ControlProblem, COCP, lqr_ce_value, psd_sqrt, QuadForm


def _beta_mean(a, b):
    return a / (a + b)


def _beta_std(a, b):
    return np.sqrt(a * b / ((a + b + 1) * (a + b) ** 2))


def _logn_mean(mu, sigma):
    return np.exp(mu + 0.5 * np.diag(sigma))


def _logn_cov(mu, sigma):
    r_mean = _logn_mean(mu, sigma)
    return (r_mean * (r_mean * (np.exp(sigma) - 1)).T).T


class CommitmentsProblem(ControlProblem):

    """Commitment optimization problem for alternative investments"""

    def __init__(
        self,
        target_nav,
        mu,
        sigma,
        call_params,
        dist_params,
        budget,
        u_reg=0.0,
        state_space_radius=1e4,
        eval_horizon=2 * 10**4,
        eval_trajectories=5,
        processes=1,
    ):
        m = len(call_params)
        assert len(dist_params) == m
        super().__init__(
            2 * m,
            m,
            1.0,
            state_space_radius,
            eval_horizon,
            eval_trajectories,
            processes,
        )

        self.target_nav = target_nav
        self.mu = mu
        self.sigma = sigma
        self.budget = budget

        self.u_reg = u_reg

        self.call_params = call_params if type(call_params) == list else [call_params]
        self.dist_params = dist_params if type(dist_params) == list else [dist_params]

        self.gamma_call_mean = np.array([_beta_mean(*p) for p in self.call_params])
        self.gamma_dist_mean = np.array([_beta_mean(*p) for p in self.dist_params])
        self.gamma_call_std = np.array([_beta_std(*p) for p in self.call_params])
        self.gamma_dist_std = np.array([_beta_std(*p) for p in self.dist_params])

        self.r_mean = _logn_mean(self.mu, self.sigma)
        self.r_cov = _logn_cov(self.mu, self.sigma)
        self.A_mean = CommitmentsProblem.form_A(
            self.r_mean, self.gamma_call_mean, self.gamma_dist_mean
        )
        self.B = np.vstack((np.zeros((m, m)), np.eye(m)))

        if self.u_reg > 0:
            x_ss, u_ss = self.create_policy().steady_state()
            self.x_ss = x_ss
            self.u_ss = u_ss

    @staticmethod
    def form_A(r, gamma_call, gamma_dist):
        m = len(gamma_call)
        return np.block(
            [
                [np.diag(r - r * gamma_dist), np.diag(gamma_call)],
                [np.diag(np.zeros(m)), np.eye(m) - np.diag(gamma_call)],
            ]
        )

    def sample_params(self):
        r = np.exp(np.random.multivariate_normal(self.mu, self.sigma))
        gamma_call = np.array([np.random.beta(*p) for p in self.call_params])
        gamma_dist = np.array([np.random.beta(*p) for p in self.dist_params])
        return r, gamma_call, gamma_dist

    def step(self, x, u):
        r, gamma_call, gamma_dist = self.sample_params()
        A = CommitmentsProblem.form_A(r, gamma_call, gamma_dist)

        x_next = A @ x + self.B @ u
        y = {"r": r, "gamma_call": gamma_call, "gamma_dist": gamma_dist}
        stage_cost = ((x[: self.m] - self.target_nav) ** 2).sum()
        if self.u_reg > 0:
            stage_cost += self.u_reg * ((u - self.u_ss) ** 2).sum()
        return x_next, y, stage_cost

    def sample_initial_condition(self):
        """Zero initial condition"""
        return np.zeros(self.n)

    def sample_random_control(self, x, t):
        """For dithering"""
        return self.budget * np.random.rand(self.m)

    def create_policy(
        self,
        lookahead=1,
        V=None,
        name="",
        compile=False,
        **kwargs,
    ):
        """Create a lookahead CommitmentsPolicy"""
        r_cov = _logn_cov(self.mu, self.sigma)
        cov_call_mat = np.block(
            [
                [np.zeros((self.m, self.m)), np.diag(self.gamma_call_std**2)],
                [np.zeros((self.m, self.m)), -np.diag(self.gamma_call_std**2)],
            ]
        )

        cov = {"A": {}}
        for i in range(self.n):
            for j in range(self.n):
                cov["A"][i, j] = np.zeros((self.n, self.n))
                if i < self.m and j < self.m:
                    # top left
                    cov["A"][i, j][i, j] = (
                        r_cov[i, j] + (i == j) * self.gamma_dist_std[i] ** 2
                    )
                elif i < self.m and j >= self.m:
                    # bottom left
                    cov["A"][i, j][:, i + self.m] = cov_call_mat[:, i + self.m]
                elif i >= self.m and j < self.m:
                    # top right
                    cov["A"][i, j][:, i] = cov_call_mat[:, i]
                elif i >= self.m and j >= self.m:
                    # bottom right
                    cov["A"][i, j][:, i] = -cov_call_mat[:, i]
        self.cov = cov

        return CommitmentsPolicy(
            self.n,
            self.m,
            gamma=self.gamma,
            lookahead=lookahead,
            V=V if V is not None else QuadForm.zero(self.n),
            A=self.A_mean,
            B=self.B,
            c=np.zeros(self.n),
            samples=1,
            cov=cov,
            target_nav=self.target_nav,
            budget=self.budget,
            name=name,
            compile=compile,
            **kwargs,
        )

    def V_lb(self, lb_slope=0.0):
        # quadratic cost on state
        Q = np.eye(self.n)
        Q[self.m :, self.m :] *= 0
        q = np.hstack((-self.target_nav, np.zeros(self.m)))

        # bound constraint 0 <= u <= B
        R = lb_slope * np.eye(self.m)
        r = -0.5 * lb_slope * self.budget * np.ones(self.m)

        # input regularization
        if self.u_reg > 0 and self.u_ss is not None:
            R += self.u_reg * np.eye(self.m)
            r += -self.u_reg * self.u_ss

        return lqr_ce_value(
            self.A_mean,
            self.B,
            Q,
            R,
            0 * np.eye(self.n),
            gamma=self.gamma,
            q=q,
            r=r,
        )

    @staticmethod
    def create_problem_instance(
        m,
        seed=0,
        eval_horizon=2 * 10**4,
        eval_trajectories=5,
        processes=1,
    ):
        """m illiquid assets"""
        if seed is not None:
            np.random.seed(seed)

        n = 2 * m

        # annual return 20%, standard deviation 30%
        mu = 0.04 + (2 * np.random.rand(m) - 1) * 0.01
        var = 0.015 + (2 * np.random.rand(m) - 1) * 0.01

        # correlation matrix
        W = (2 * np.random.rand(m, m) - 1) * 10
        # W = np.random.rand(m, m)
        corr = W @ W.T + np.diag(np.random.rand(m))
        d = np.diag(1 / np.sqrt(np.diag(corr)))
        corr = d @ corr @ d

        # sigma matrix
        sigma = np.diag(np.sqrt(var)) @ corr @ np.diag(np.sqrt(var))

        call_params = [
            (2, 12 + (2 * np.random.rand() - 1) * 3) for _ in range(m)
        ]  # mean is 0.1429
        dist_params = [
            (3, 15 + (2 * np.random.rand() - 1) * 3) for _ in range(m)
        ]  # mean is 0.1667

        # problem parameters
        target_nav = 4 + np.random.rand(m)
        budget = 3
        u_reg = 1e-2

        return CommitmentsProblem(
            target_nav,
            mu,
            sigma,
            call_params,
            dist_params,
            budget=budget,
            u_reg=u_reg,
            state_space_radius=1e4,
            eval_horizon=eval_horizon,
            eval_trajectories=eval_trajectories,
            processes=processes,
        )


class CommitmentsPolicy(COCP):
    def stage_cost(self, x, u):
        budget = self.params["budget"]
        target_nav = self.params["target_nav"]
        cost = cp.sum_squares(x[: self.m] - target_nav)
        if "u_reg" in self.params and "u_ss" in self.params:
            u_reg = self.params["u_reg"]
            if u_reg > 0:
                u_ss = self.params["u_ss"]
                cost += u_reg * cp.sum_squares(u - u_ss)
        constraints = [u >= 0, u <= budget]
        return cost, constraints

    def update_value(self, V):
        """Update parameters with new value function"""
        self.V = V
        if V is None:
            H = np.zeros((self.n + self.m, self.n + self.m))
            h = np.zeros(self.n + self.m)
            const = 0
        else:
            P, p, pi = V.P, V.p, V.c
            A, B, c = self.A, self.B, self.c

            M11 = A.T @ P @ A
            for i in range(P.shape[0]):
                for j in range(P.shape[0]):
                    M11[i, j] += np.trace(P @ self.params["cov"]["A"][i, j])

            M12 = A.T @ P @ B
            M13 = (A.T @ P @ c + A.T @ p).reshape(-1, 1)
            M22 = B.T @ P @ B
            M23 = (B.T @ P @ c + B.T @ p).reshape(-1, 1)
            M33 = (c.T @ P @ c + 2 * c.T @ p + pi).reshape(1, 1)
            M = np.block([[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]])

            H = M[: self.n + self.m, : self.n + self.m]
            h = M[-1, :-1]
            const = (self.gamma**self.lookahead) * M[-1, -1]

        self.update_params(H_sqrt=psd_sqrt(H), h=h, const=const)


def commitment_cvxpylayer(problem):
    """Create a cvxpylayer policy for the commitment problem"""
    n, m = problem.n, problem.m
    gamma = problem.gamma

    target_nav = problem.target_nav
    budget = problem.budget
    u_reg = problem.u_reg
    u_ss = problem.u_ss

    x = cp.Parameter((n, 1))
    H_sqrt = cp.Parameter((n + m, n + m))
    h = cp.Parameter(n + m)

    xu = cp.Variable((n + m, 1))
    u = cp.Variable((m, 1))

    stage_cost = cp.sum_squares(xu[:m, 0] - target_nav)
    if u_reg > 0:
        stage_cost += cp.sum_squares(u[:, 0] - u_ss)
    value_next = cp.sum_squares(H_sqrt @ xu) + 2 * h.T @ xu
    constraints = [xu == cp.vstack((x, u)), u >= 0, u <= budget]
    prob = cp.Problem(cp.Minimize(stage_cost + gamma * value_next), constraints)
    policy = CvxpyLayer(prob, [x, H_sqrt, h], [u])
    return policy


def form_H_sqrt_and_h(problem, P_sqrt, p):
    """Get parameters of EV(Ax+Bu+c) for commitments"""
    A, B = map(torch.from_numpy, (problem.A_mean, problem.B))

    # form M, H, h
    P = P_sqrt.T @ P_sqrt
    M11 = A.T @ P @ A
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            M11[i, j] += torch.trace(P @ torch.from_numpy(problem.cov["A"][i, j]))

    M12 = A.T @ P @ B
    M13 = (A.T @ p).reshape(-1, 1)
    M22 = B.T @ P @ B
    M23 = (B.T @ p).reshape(-1, 1)
    M33 = torch.zeros((1, 1))

    M1 = torch.cat((M11, M12, M13), dim=1)
    M2 = torch.cat((M12.T, M22, M23), dim=1)
    M3 = torch.cat((M13.T, M23.T, M33), dim=1)
    M = torch.cat((M1, M2, M3), dim=0)
    H = M[: problem.n + problem.m, : problem.n + problem.m]
    h = M[-1, :-1]

    # form H_sqrt
    eval, evec = torch.linalg.eigh(H)
    H_sqrt = torch.diag(torch.sqrt(torch.clamp(eval, min=0.0, max=None))) @ (evec.T)
    return H_sqrt, h


def commitment_cocp_grad(
    problem,
    samples_per_iter,
    num_iters,
    learning_rate,
    num_trajectories=1,
    seed=None,
    V0=None,
    Vlb=None,
    policy=None,
    eval_freq=None,
    lambda_l2=0.0,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="little")
    torch.manual_seed(seed)
    np.random.seed(seed)

    eval_freq = num_iters if eval_freq is None else eval_freq

    B, target_nav, u_ss = map(
        torch.from_numpy, (problem.B, problem.target_nav, problem.u_ss)
    )
    torch_policy = commitment_cvxpylayer(problem)

    # loss function
    def lossf(time_horizon, batch_size, P_sqrt, q, x_batch=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # parameters
        B_batch = B.repeat(batch_size, 1, 1)
        target_nav_batch = target_nav.repeat(batch_size, 1).unsqueeze(-1).double()
        u_ss_batch = u_ss.repeat(batch_size, 1).unsqueeze(-1).double()

        # form H_sqrt and h
        H_sqrt, h = form_H_sqrt_and_h(problem, P_sqrt, q)

        # initial condition
        if x_batch is None:
            x_batch = (
                torch.tensor(
                    np.vstack(
                        [problem.sample_initial_condition() for _ in range(batch_size)]
                    )
                )
                .double()
                .unsqueeze(-1)
            )

        # propagate forward
        loss = 0.0
        for t in range(time_horizon):
            u_batch = torch_policy(x_batch, H_sqrt, h)[0]

            tracking_cost = torch.sum(
                (x_batch[:, : problem.m, :] - target_nav_batch) ** 2
            )
            regularization = problem.u_reg * torch.sum((u_batch - u_ss_batch) ** 2)
            loss += (tracking_cost + regularization) / (batch_size * time_horizon)

            # sample A matrix
            r, gamma_call, gamma_dist = problem.sample_params()
            A = torch.from_numpy(CommitmentsProblem.form_A(r, gamma_call, gamma_dist))
            A_batch = A.repeat(batch_size, 1, 1).double()

            x_batch = (
                torch.bmm(A_batch, x_batch) + torch.bmm(B_batch, u_batch)
            ).double()

        if lambda_l2 > 0.0:
            loss += lambda_l2 * (torch.sum(P_sqrt**2) + 2 * torch.sum(q**2))

        return loss, x_batch.detach()

    # initialize parameters
    if V0 is None:
        P_sqrt = torch.eye(problem.n).double()
        P_sqrt.requires_grad_(True)
        p = torch.zeros(problem.n).double()
        p.requires_grad_(True)
    else:
        P_sqrt = torch.from_numpy(psd_sqrt(V0.P))
        P_sqrt.requires_grad_(True)
        p = torch.from_numpy(V0.p).double()
        p.requires_grad_(True)

    # run optimization
    opt = torch.optim.Adam([P_sqrt, p], lr=learning_rate)
    costs = []
    value_iterates = []
    x0 = None
    for k in range(num_iters):
        # evaluate
        with torch.no_grad():
            U = P_sqrt.detach().numpy()
            V = QuadForm(U, p.detach().numpy(), 0.0)
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
                    p.detach(),
                    seed=0,
                )[0].item()
                costs.append(expected_cost)
                print(
                    "it: %03d, test loss: %3.3f, policy cost: %3.3f"
                    % (k + 1, test_loss, expected_cost)
                )

        # gradient step
        opt.zero_grad()
        l, x0 = lossf(
            samples_per_iter,
            num_trajectories,
            P_sqrt,
            p,
            x_batch=None,
            seed=seed + k + 1,
        )
        l.backward()
        opt.step()

        # enforce lower bound
        if Vlb is not None:
            P_sqrt_npy = P_sqrt.detach().numpy()

            P_curr = P_sqrt_npy.T @ P_sqrt_npy
            p_curr = p.detach().numpy()

            _P = cp.Variable((problem.n, problem.n), PSD=True)
            _p = cp.Variable((problem.n, 1))
            _pi = cp.Variable((1, 1))
            block = cp.bmat([[_P, _p], [_p.T, _pi]])
            block_lb = np.block(
                [[Vlb.P, Vlb.p.reshape(-1, 1)], [Vlb.p.reshape(1, -1), Vlb.pi]]
            )
            proj = cp.Problem(
                cp.Minimize(
                    cp.sum_squares(_P - P_curr) + 2 * cp.sum_squares(_p[:, 0] - p_curr)
                ),
                [block - block_lb >> 0],
            )
            proj.solve()

            P_sqrt.data.fill_(0)
            P_sqrt.data += torch.from_numpy(psd_sqrt(_P.value))
            p.data.fill_(0)
            p.data += torch.from_numpy(_p.value).squeeze()

    return {"costs": costs, "iterates": value_iterates}
