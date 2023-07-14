import numpy as np
import cvxpy as cp
import os
import torch
from cvxpylayers.torch import CvxpyLayer

from scipy.sparse import lil_matrix, csr_matrix, bmat, vstack, identity, diags
from vgi import ControlProblem, COCP, lqr_ce_value, psd_sqrt, QuadForm


class SupplyChainProblem(ControlProblem):

    """Supply chain optimization problem"""

    def __init__(
        self,
        num_nodes,
        buyers,
        sellers,
        transports,
        h_max,
        u_max,
        alpha,
        beta,
        tau,
        r,
        mu_buy_price,
        sigma_buy_price,
        mu_demand,
        sigma_demand,
        state_space_radius=1e4,
        eval_horizon=2 * 10**4,
        eval_trajectories=5,
        processes=1,
    ):
        # buyer and seller node indices
        self.buyers = buyers
        self.sellers = sellers
        self.nodes = [i for i in range(num_nodes) if i not in buyers + sellers]

        # list of transport edge tuples
        self.transports = transports

        # number of warehouses
        self.num_nodes = num_nodes

        # problem dimensions
        self.num_buyers = len(self.buyers)
        self.num_sellers = len(self.sellers)
        self.num_transports = len(self.transports)

        m = self.num_buyers + self.num_sellers + self.num_transports
        super().__init__(
            num_nodes,
            m,
            1.0,
            state_space_radius,
            eval_horizon,
            eval_trajectories,
            processes,
        )

        # capacity constraints
        self.h_max = h_max
        self.u_max = u_max

        # storage cost parameters
        self.alpha = alpha
        self.beta = beta

        # transportation cost parameter
        self.tau = tau

        # fixed sell prices
        self.r = r

        # generate incidence matrices
        self.A_in, self.A_out = SupplyChainProblem.edges_to_matrix(
            self.num_nodes, self.buyers, self.sellers, self.transports
        )
        self.B = self.A_in - self.A_out

        # generate A matrix
        self.A = identity(num_nodes)

        # log normal statistics
        logn_mean = lambda mu, sigma: np.exp(mu + 0.5 * sigma**2)
        logn_var = lambda mu, sigma: (np.exp(sigma**2) - 1) * np.exp(
            2 * mu + sigma**2
        )

        # buy prices
        self.mu_buy_price = mu_buy_price
        self.sigma_buy_price = sigma_buy_price
        self.mean_buy_price = logn_mean(self.mu_buy_price, self.sigma_buy_price)
        self.var_buy_price = logn_var(self.mu_buy_price, self.sigma_buy_price)

        # demands
        self.mu_demand = mu_demand
        self.sigma_demand = sigma_demand
        self.mean_demand = logn_mean(self.mu_demand, self.sigma_demand)
        self.var_demand = logn_var(self.mu_demand, self.sigma_demand)

    @staticmethod
    def edges_to_matrix(n, buyers, sellers, transports):
        """Convert edge list to dynamics matrices
        buyers: list of buyer nodes
        sellers: list of seller nodes
        transports: internode transport edge list
        """
        m = len(buyers) + len(sellers) + len(transports)
        A_in = lil_matrix((n, m))
        A_out = lil_matrix((n, m))

        for i in range(len(buyers)):
            A_in[buyers[i], i] = 1

        for i in range(len(sellers)):
            A_out[sellers[i], len(buyers) + i] = 1

        for k in range(len(transports)):
            i, j = transports[k]
            A_in[j, len(buyers) + len(sellers) + k] = 1
            A_out[i, len(buyers) + len(sellers) + k] = 1

        return A_in.tocsr(), A_out.tocsr()

    def sample_price_demand(self):
        p = np.exp(
            np.random.multivariate_normal(
                self.mu_buy_price, np.diag(self.sigma_buy_price**2)
            )
        )
        d = np.exp(
            np.random.multivariate_normal(
                self.mu_demand, np.diag(self.sigma_demand**2)
            )
        )
        return p, d

    def step(self, x, u):
        # unpack state
        h = x[: self.num_nodes]
        p = x[self.num_nodes : self.num_nodes + self.num_buyers]

        # unpack control
        b = u[: self.num_buyers]
        s = u[self.num_buyers : self.num_buyers + self.num_sellers]
        z = u[self.num_buyers + self.num_sellers :]

        # stage cost
        stage_cost = -self.r @ s + self.tau @ z
        stage_cost += self.alpha @ h + self.beta @ (h**2)
        stage_cost += p @ b

        # dynamics
        p, d = self.sample_price_demand()
        h_next = np.clip(h + self.B @ u, 0, self.h_max)
        x_next = np.hstack((h_next, p, d))
        return x_next, np.hstack((p, d)), stage_cost

    def sample_initial_condition(self):
        h = self.h_max * np.random.rand(self.num_nodes)
        p, d = self.sample_price_demand()
        return np.hstack((h, p, d))

    def sample_random_control(self, x, t):
        # TODO: implement a reference policy for control_problem
        # and then use u.project() to implement this method
        # unpack state
        h = x[: self.num_nodes]
        d = x[self.num_nodes + self.num_buyers :]

        # generate random control
        u_candidate = self.u_max * np.random.rand(self.m)

        # make sure that the control is feasible
        u = cp.Variable(self.m)
        s = u[self.num_buyers : self.num_buyers + self.num_sellers]
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(u - u_candidate)),
            [
                self.A_out @ u <= h,
                u >= 0,
                u <= self.u_max,
                h + self.B @ u >= 0,
                h + self.B @ u <= self.h_max,
                s <= d,
            ],
        )
        prob.solve()
        return u.value

    def create_policy(
        self,
        lookahead=1,
        V=None,
        name="",
        compile=False,
        **kwargs,
    ):
        stage_cost_params = {
            "p": self.mean_buy_price,
            "d": self.mean_demand,
        }
        return SupplyChainPolicy(
            self.n,
            self.m,
            gamma=self.gamma,
            lookahead=lookahead,
            V=V if V is not None else QuadForm.zero(self.n),
            A=np.eye(self.n),
            B=self.B.toarray(),
            c=np.zeros(self.n),
            alpha=self.alpha,
            beta=self.beta,
            r=self.r,
            tau=self.tau,
            A_in=self.A_in,
            A_out=self.A_out,
            u_max=self.u_max,
            h_max=self.h_max,
            num_buyers=self.num_buyers,
            num_sellers=self.num_sellers,
            name=name,
            compile=compile,
            stage_cost_params=stage_cost_params,
            **kwargs,
        )

    def V_lb(self, lb_slope=1.0):
        n = self.n + self.num_buyers

        A = bmat(
            [
                [identity(self.num_nodes), None],
                [None, csr_matrix((self.num_buyers, self.num_buyers))],
            ]
        ).toarray()
        B = vstack((self.B, csr_matrix((self.num_buyers, self.m)))).toarray()

        Q = np.zeros((n, n))
        Q[: self.n, : self.n] = np.diag(self.beta)
        q = 0.5 * np.hstack((self.alpha, np.zeros(self.num_buyers)))

        R = np.zeros((self.m, self.m))
        r = 0.5 * np.hstack((np.zeros(self.num_buyers), -self.r, self.tau))

        S = np.zeros((n, self.m))
        S[-self.num_buyers :, : self.num_buyers] = 0.5 * np.eye(self.num_buyers)

        w_mean = np.hstack((np.zeros(self.n), self.mean_buy_price))
        Cov = np.zeros((n, n))
        Cov[-self.num_buyers :, -self.num_buyers :] = np.diag(self.var_buy_price)

        # lower bound on input box constraint
        R += lb_slope * np.eye(self.m)
        r += (
            -0.5
            * lb_slope
            * self.u_max
            * (np.ones(self.m) if type(self.u_max) == float else self.u_max)
        )
        Vlb = lqr_ce_value(A, B, Q, R, Cov, gamma=1.0, q=q, r=r, S=S, w_mean=w_mean)
        return QuadForm(P=Vlb.P[: self.n, : self.n], p=Vlb.p[: self.n], c=Vlb.c)

    @staticmethod
    def create_problem_instance(
        eval_horizon=2 * 10**4,
        eval_trajectories=5,
        processes=1,
    ):
        n_source = 2
        n_sink = 2

        num_nodes = 4
        source_nodes = [0, 1]
        sink_nodes = [2, 3]
        edge_list = [(0, 2), (0, 3), (1, 3), (3, 2)]
        h_max = 3.0
        u_max = 2.0
        alpha = 0.01 * np.ones(num_nodes)
        beta = 0.01 * np.ones(num_nodes)
        tau = 0.05 * np.ones(len(edge_list))
        r = 1.3 * np.ones(n_sink)

        mu_buy_price = np.array([0.0, 0.1])
        sigma_buy_price = 0.4 * np.ones(n_source)
        mu_demand = np.array([0.0, 0.4])
        sigma_demand = 0.4 * np.ones(n_sink)

        return SupplyChainProblem(
            num_nodes,
            source_nodes,
            sink_nodes,
            edge_list,
            h_max,
            u_max,
            alpha,
            beta,
            tau,
            r,
            mu_buy_price,
            sigma_buy_price,
            mu_demand,
            sigma_demand,
            state_space_radius=1e4,
            eval_horizon=eval_horizon,
            eval_trajectories=eval_trajectories,
            processes=processes,
        )


class SupplyChainPolicy(COCP):
    def __call__(self, x, t):
        """Evaluate the policy at augmented state x"""
        num_buyers = self.params["num_buyers"]

        # unpack state
        h = x[: self.n]
        p = x[self.n : self.n + num_buyers]
        d = x[self.n + num_buyers :]

        return super().__call__(h, t, **{"p": p, "d": d})

    def stage_cost(self, x, u, p=None, d=None):
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        r = self.params["r"]
        tau = self.params["tau"]

        A_in = self.params["A_in"]
        A_out = self.params["A_out"]
        u_max = self.params["u_max"]
        h_max = self.params["h_max"]

        num_buyers = self.params["num_buyers"]
        num_sellers = self.params["num_sellers"]

        # unpack state
        h = x[: self.n]
        # unpack input
        b = u[:num_buyers]
        s = u[num_buyers : num_buyers + num_sellers]
        z = u[num_buyers + num_sellers :]

        # stage cost
        stage_cost = -r @ s + tau @ z
        stage_cost += alpha @ h + beta @ cp.square(h)
        stage_cost += p @ b

        constraints = [
            A_out @ u <= h,
            u >= 0,
            u <= u_max,
            h + (A_in - A_out) @ u >= 0,
            h + (A_in - A_out) @ u <= h_max,
            s <= d,
        ]

        return stage_cost, constraints


def supply_chain_cvxpylayer(problem):
    """Create a cvxpylayer policy for the supply chain problem"""
    n = problem.n
    num_buyers = problem.num_buyers
    num_sellers = problem.num_sellers
    num_transports = problem.num_transports
    gamma = problem.gamma
    h_max = problem.h_max
    u_max = problem.u_max
    alpha = problem.alpha
    beta = problem.beta
    tau = problem.tau
    r = problem.r

    # cvxpy problem
    P_sqrt = cp.Parameter((n, n))
    q = cp.Parameter(n)

    b = cp.Variable((num_buyers, 1))
    s = cp.Variable((num_sellers, 1))
    z = cp.Variable((num_transports, 1))
    u = cp.vstack((b, s, z))
    x_next = cp.Variable((n, 1))

    x = cp.Parameter((n, 1))
    p = cp.Parameter((num_buyers, 1))
    d = cp.Parameter((num_sellers, 1))

    # stage cost
    stage_cost = (
        cp.vstack((p, -r.reshape(-1, 1), tau.reshape(-1, 1))).T @ u
        + alpha @ x
        + beta @ (x**2)
    )
    value_next = gamma * (cp.sum_squares(P_sqrt @ x_next) + 2 * q.T @ x_next)
    constraints = [
        x_next == x + (problem.A_in - problem.A_out) @ u,
        x_next >= 0,
        x_next <= h_max,
        u >= 0,
        u <= u_max,
        problem.A_out @ u <= x,
        s <= d,
    ]
    prob = cp.Problem(cp.Minimize(stage_cost + value_next), constraints)
    policy = CvxpyLayer(prob, [x, p, d, P_sqrt, q], [b, s, z])
    return policy


def supply_chain_cocp_grad(
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
    n = problem.num_nodes
    B, r, tau, alpha, beta = map(
        torch.from_numpy,
        [
            problem.A_in_out.toarray(),
            problem.r,
            problem.tau,
            problem.alpha,
            problem.beta,
        ],
    )
    alpha = alpha.reshape(1, -1)
    tau = tau.reshape(1, -1)
    r = r.reshape(1, -1)
    torch_policy = supply_chain_cvxpylayer(problem)

    # define loss
    def lossf(time_horizon, batch_size, P_sqrt, q, x_batch=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        # x_batch: (batch size, n, 1)
        if x_batch is None:
            x_batch = (
                torch.tensor(
                    np.vstack(
                        [
                            problem.sample_initial_condition()[: problem.num_nodes]
                            for _ in range(batch_size)
                        ]
                    )
                )
                .double()
                .unsqueeze(-1)
            )

        # price and demand
        price, demand = [], []
        for _ in range(time_horizon * batch_size):
            pr, de = problem.sample_price_demand()
            price.append(pr)
            demand.append(de)

        price_batch = (
            torch.tensor(np.vstack(price))
            .double()
            .reshape(batch_size, -1, time_horizon)
        )
        demand_batch = (
            torch.tensor(np.vstack(demand))
            .double()
            .reshape(batch_size, -1, time_horizon)
        )

        P_sqrt_batch = P_sqrt.repeat(batch_size, 1, 1)
        q_batch = q.repeat(batch_size, 1)

        B_batch = B.repeat(batch_size, 1, 1)
        r_batch = r.repeat(batch_size, 1, 1)
        tau_batch = tau.repeat(batch_size, 1, 1)
        alpha_batch = alpha.repeat(batch_size, 1, 1)
        beta_batch = torch.diag(beta).repeat(batch_size, 1, 1)
        loss = 0.0
        for t in range(time_horizon):
            buy_batch, sell_batch, transport_batch = torch_policy(
                x_batch,
                price_batch[:, :, t].unsqueeze(-1),
                demand_batch[:, :, t].unsqueeze(-1),
                P_sqrt_batch,
                q_batch,
                solver_args={"solve_method": "ECOS"},
            )
            u_batch = torch.cat((buy_batch, sell_batch, transport_batch), dim=1)

            holding_cost = torch.bmm(
                torch.bmm(beta_batch, x_batch).transpose(2, 1), x_batch
            )
            holding_cost += torch.bmm(alpha_batch, x_batch)

            transport_cost = torch.bmm(tau_batch, transport_batch)
            revenue = torch.bmm(r_batch, sell_batch)
            material_cost = torch.bmm(
                price_batch[:, :, t].unsqueeze(-1).transpose(-1, -2), buy_batch
            )

            cost_batch = holding_cost.squeeze()
            cost_batch += transport_cost.squeeze()
            cost_batch -= revenue.squeeze()
            cost_batch += material_cost.squeeze()

            loss += cost_batch.mean() / (time_horizon * batch_size)

            x_batch = (x_batch + torch.bmm(B_batch, u_batch)).double()

        if lambda_l2 > 0.0:
            loss += lambda_l2 * (torch.sum(P_sqrt**2) + 2 * torch.sum(q**2))

        return loss, x_batch.detach()

    # initialize parameters
    if V0 is None:
        P_sqrt = torch.eye(n).double()
        P_sqrt.requires_grad_(True)
        p = -problem.h_max * torch.ones(n).double()
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

        opt.zero_grad()
        l, x0 = lossf(
            samples_per_iter, num_trajectories, P_sqrt, p, x_batch=None, seed=k + 1
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
