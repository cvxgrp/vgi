import numpy as np


class Policy:
    def compile(self, policy_name):
        pass

    def update_params(self, V, dynamics):
        pass

    def clone(self):
        raise NotImplementedError

    def __call__(self, x, t):
        raise NotImplementedError

    @property
    def value(self):
        """Get value for fitting value function"""
        return np.nan

    @property
    def value_gradient(self):
        """Get value gradient for fitting value function"""
        return np.nan


class AffinePolicy(Policy):
    """Affine policy has form u = K @ x + k"""

    def __init__(self, K, k):
        self.K = K
        self.k = k

    def clone(self):
        return AffinePolicy(self.K.copy(), self.k.copy())

    def __call__(self, x, t):
        return self.K @ x + self.k
