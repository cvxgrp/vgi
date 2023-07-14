import numpy as np
import os
from .quad_form import QuadForm


class FVI:
    """Fitted value iteration"""

    def __init__(
        self,
        control_problem,
        policy,
        value_fitter,
        damping=0.5,
        trajectory_len=50,
        num_trajectories=1,
        V0=None,
        restart_simulations=False,
    ):
        self.control_problem = control_problem
        self.policy = policy.clone()
        self.value_fitter = value_fitter
        self.damping = damping
        self.trajectory_len = trajectory_len
        self.num_trajectories = num_trajectories
        self.restart_simulations = restart_simulations

        # initial value function
        if V0 is None:
            self.V0 = QuadForm.eye(control_problem.n)
        else:
            self.V0 = V0.clone()

    def __call__(
        self, num_iters, eval_freq=None, V0=None, verbose=True, seed=None, dithering=0.0
    ):
        seed = (
            int.from_bytes(os.urandom(4), byteorder="little") if seed is None else seed
        )
        np.random.seed(seed)

        V = self.V0.clone() if V0 is None else V0.clone()
        policy = self.policy.clone()
        policy.update_value(V)

        self.iterates = [
            V.clone(),
        ]
        self.step_sizes = []
        self.fitting_scores = []
        self.costs = []
        if verbose and eval_freq is not None:
            self.costs.append(self.evaluate_iterate(policy, seed))

        x0_list = [None for _ in range(self.num_trajectories)]
        for i in range(1, num_iters + 1):
            # simulate current policy to get data
            sims = self.generate_data(
                policy, x0_list, self.trajectory_len, seed + i, dithering
            )
            X, y = self.extract_data(sims)

            # remember last state of each trajectory
            if self.control_problem.gamma == 1 and not self.restart_simulations:
                x0_list = [
                    sims[j].states_matrix[-1] for j in range(self.num_trajectories)
                ]

            # fit value function
            self.value_fitter.fit(X, y)
            fitting_score = self.value_fitter.score(X, y)

            # form update with damping
            alpha = 1 - self.damping
            TV = self.value_fitter.V_.clone()
            V_next = QuadForm.linear_combination((alpha, 1 - alpha), (TV, V))

            # update data
            self.iterates.append(V_next.clone())
            self.step_sizes.append(np.linalg.norm(V_next.params - V.params))
            self.fitting_scores.append(fitting_score)
            if verbose and eval_freq is not None and i % eval_freq == 0:
                self.costs.append(self.evaluate_iterate(policy, seed + num_iters + i))

            # update policy and value function
            policy.update_value(V_next.clone())
            V = V_next.clone()

            if verbose:
                if i == 1:
                    # print header
                    header = "    It.\td(TV, V) \tdamping\t\regression score"
                    if verbose and len(self.costs) > 0:
                        header += "\t\tcost"
                    print(header)

                message = "%d\t%0.2e\t%0.2e\t%0.2e" % (
                    i,
                    self.step_sizes[-1],
                    self.damping,
                    fitting_score,
                )
                if verbose and eval_freq is not None and (i == 1 or i % eval_freq == 0):
                    message += "\t%0.2e" % self.costs[-1]
                print(message)

        return policy

    def evaluate_iterate(self, policy, seed):
        """evaluate cost of policy"""
        return self.control_problem.cost(policy.clone(), seed=seed)

    def generate_data(self, policy, x0_list, sim_length, seed, dithering=0.0):
        """generate list of simulations"""
        return [
            self.control_problem.simulate(
                policy.clone(),
                sim_length,
                x0=x0,
                seed=seed + sim_length * len(x0_list) * idx,
                dithering=dithering,
            )
            for idx, x0 in enumerate(x0_list)
        ]

    def extract_data(self, sim_list):
        """extract training data from list of simulations
        if state dimension exceeds that of policy, then only extract
        the first n state dimensions
        """
        X = np.vstack([sim.states_matrix for sim in sim_list])
        y = np.hstack([sim.bellman_values for sim in sim_list])
        return X[:, : self.policy.n], y


class VGI(FVI):
    """Value-gradient iteration"""

    def extract_data(self, sim_list):
        """extract training data from list of simulations
        if state dimension exceeds that of policy, then only extract
        the first n state dimensions
        """
        X = np.vstack([sim.states_matrix for sim in sim_list])
        y = np.vstack([sim.bellman_value_gradients for sim in sim_list])
        return X[:, : self.policy.n], y[:, : self.policy.n]
