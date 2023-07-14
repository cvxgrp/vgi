import numpy as np


class Simulation:
    def __init__(self, gamma=1.0):
        assert 0 <= gamma <= 1, "gamma must be in [0, 1]"
        self.gamma = gamma
        self.T = 0
        self._success = True
        self._x = []
        self._y = []
        self._u = []
        self._params = []
        self._bellman_values = []
        self._bellman_value_gradients = []
        self._stage_costs = []

    def update(self, x, y, u, stage_cost, bellman_value, bellman_value_gradient):
        """Log new state, control, stage_cost"""
        self._x.append(x)
        self._y.append(y)
        self._u.append(u)
        self._bellman_values.append(bellman_value)
        self._bellman_value_gradients.append(bellman_value_gradient)
        self._stage_costs.append(stage_cost)
        self.T += 1

    @property
    def success(self):
        """Track if simulation succeeded"""
        return self._success

    @success.setter
    def success(self, is_successful):
        self._success = is_successful

    @property
    def horizon(self):
        """Number of timesteps logged"""
        return self.T

    @property
    def states(self):
        """List of states visited"""
        return self._x

    @property
    def states_matrix(self):
        """Matrix of states visited, size (n, T)"""
        return np.vstack(self.states)

    @property
    def observations(self):
        """List of observations made"""
        return self._y

    @property
    def controls(self):
        """List of controls used"""
        return self._u

    @property
    def controls_matrix(self):
        """Matrix of controls used, size (n, T)"""
        return np.vstack(self.controls)

    @property
    def stage_costs(self):
        """List of controls applied"""
        return self._stage_costs

    @property
    def cost(self):
        """List of stage cost values"""
        if self.gamma < 1:
            return np.sum(
                [c * self.gamma**t for t, c in enumerate(self._stage_costs)]
            )
        else:
            return np.mean(self._stage_costs)

    @property
    def bellman_values(self):
        """List of computed Bellman values"""
        return self._bellman_values

    @property
    def bellman_value_gradients(self):
        """List of computed Bellman values"""
        return self._bellman_value_gradients
