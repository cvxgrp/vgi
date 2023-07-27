import numpy as np
import os
from functools import partial
from pathos.pools import _ProcessPool
import pathos.profile as pr
import multiprocess.context as ctx
from inspect import signature

from .simulation import Simulation


class ControlProblem:

    """Class for defining a control problem"""

    # global memory for multiprocessing
    policy = {}
    problems = {}

    def __init__(
        self,
        n,
        m,
        gamma=1.0,
        state_space_radius=1e4,
        eval_horizon=2 * 10**4,
        eval_trajectories=5,
        processes=1,
    ):
        self.n = n
        self.m = m

        # discount factor
        assert 0 <= gamma <= 1, "gamma must be in [0, 1]"
        self.gamma = gamma

        # threshold for detecting instability
        self.state_space_radius = state_space_radius

        # cost eval parameters
        self.eval_horizon = eval_horizon
        self.eval_trajectories = eval_trajectories
        self.processes = processes

    def step(self, x, u):
        """Dynamics for simulation. Returns a tuple (z, y, L),
        where z is next state, y is observation, and stage cost L
        """
        raise NotImplementedError

    def sample_initial_condition(self):
        """Draw a sample from model's named variable 'x0'"""
        raise NotImplementedError

    def sample_random_control(self, x, t):
        """Sample a random control at state x and time t"""
        raise NotImplementedError

    def clone(self):
        """Create a new instantiation of the same control problem"""
        params = signature(type(self).__init__).parameters
        return type(self)(**{p: getattr(self, p) for p in params.keys() if p != "self"})

    def simulate(
        self,
        policy,
        T,
        x0=None,
        seed=None,
        dithering=0.0,
        log_planned_steps=False,
    ):
        """Simulate a trajectory using a policy and return Simulation object"""
        seed = (
            int.from_bytes(os.urandom(4), byteorder="little") if seed is None else seed
        )
        np.random.seed(seed)

        policy = policy.clone()
        summary = Simulation(self.gamma)
        x = self.sample_initial_condition() if x0 is None else x0
        for t in range(T):
            # evaluate policy
            u = policy(x, t)
            if u is None:
                summary.success = False
                break

            # log value data
            bellman_value = policy.value
            bellman_value_gradient = policy.value_gradient

            # log planned states and actions in lookahead policy
            if log_planned_steps:
                planned_states = policy.planned_states
                planned_controls = policy.planned_controls
            else:
                planned_states = None
                planned_controls = None

            # dithering
            if np.random.rand() < dithering:
                u = self.sample_random_control(x, t)

            # calculate next state, observations, and stage cost
            x_next, y, stage_cost = self.step(x, u)
            summary.update(
                x,
                y,
                u,
                stage_cost,
                bellman_value,
                bellman_value_gradient,
                planned_states=planned_states,
                planned_controls=planned_controls,
            )
            x = x_next

            # check for divergence
            if np.linalg.norm(x) > self.state_space_radius * np.sqrt(self.n):
                summary.success = False
                break

        return summary

    def _cost(
        self,
        policy,
        T,
        x0=None,
        seed=None,
    ):
        """Simulate a trajectory of a policy and return the cost"""
        seed = (
            int.from_bytes(os.urandom(4), byteorder="little") if seed is None else seed
        )
        np.random.seed(seed)

        steps_taken = 0
        cost = 0.0
        x = self.sample_initial_condition() if x0 is None else x0
        for t in range(T):
            # evaluate policy
            u = policy(x, t)
            if u is None:
                cost = np.nan
                break

            # calculate next state and stage cost
            x, _, stage_cost = self.step(x, u)
            cost += (self.gamma**t) * stage_cost
            steps_taken += 1

            # check for divergence
            if np.linalg.norm(x) > self.state_space_radius * np.sqrt(self.n):
                cost = np.inf
                break

        # handle average cost case
        if self.gamma == 1.0 and steps_taken > 0:
            cost /= steps_taken

        return cost

    def cost(
        self,
        policy,
        x0=None,
        seed=None,
        reduction=np.mean,
    ):
        """Evaluate expected cost of a policy
        starting from state x0 (may be None,
        in which case a random initial state is used)"""
        base_seed = (
            int.from_bytes(os.urandom(4), byteorder="little") if seed is None else seed
        )
        if self.processes > 1:
            ctx._force_start_method("spawn")
            with _ProcessPool(
                processes=self.processes,
                initializer=ControlProblem._init_sim_worker,
                initargs=(self, policy),
            ) as pool:
                results = list(
                    pool.map_async(
                        partial(
                            ControlProblem._cost_worker, x0=x0, T=self.eval_horizon
                        ),
                        [base_seed + i for i in range(self.eval_trajectories)],
                    ).get()
                )
                pool.close()
                pool.join()
        else:

            def trajectory_cost(seed):
                return self._cost(policy.clone(), self.eval_horizon, x0=x0, seed=seed)

            results = list(
                map(
                    trajectory_cost,
                    [base_seed + i for i in range(self.eval_trajectories)],
                )
            )

        return results if reduction is None else reduction(results)

    # internal methods for simulation and cost evaluation
    @staticmethod
    def _init_sim_worker(problem, policy):
        """Initialize a worker for running parallel simulations, with policy and ControlProblem objects"""
        pid = pr.process_id()
        ControlProblem.policy[pid] = policy.clone()
        ControlProblem.problems[pid] = problem.clone()

    @staticmethod
    def _cost_worker(seed, x0, T):
        pid = pr.process_id()
        policy = ControlProblem.policy[pid]
        problem = ControlProblem.problems[pid]
        cost = problem._cost(policy, T, x0=x0, seed=seed)
        return cost
