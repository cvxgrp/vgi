# VGI - A method for convex stochastic control
[![Main Test](https://github.com/cvxgrp/vgi/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/cvxgrp/vgi/actions/workflows/test.yml)

Value-gradient iteration (VGI) is a method for designing policies for convex stochastic control problems
characterized by random linear dynamics and convex stage cost. We consider policies
that employ quadratic approximate value functions as a substitute for the true value
function. Evaluating the associated control policy involves solving a convex problem,
typically a quadratic program, which can be carried out reliably in real-time. VGI fits 
the gradient of the value function with regularization that can include
constraints reflecting known bounds on the true value function. Our value-gradient
iteration method can yield a good approximate value function with few samples, and
little hyperparameter tuning.

For more details, see our [manuscript](https://stanford.edu/~boyd/papers/pdf/vgi.pdf).

To install locally, clone the repository, and run
```pip install -e .```
in the repo directory. Optionally, create a pyenv or conda environment first. Note that the examples require additional dependencies,[```torch```](https://pytorch.org/) and [```cvxpylayers```](https://github.com/cvxgrp/cvxpylayers).

## Convex stochastic control
We consider convex stochastic control problems, which have dynamics
$$x_{t+1} = A_tx_t + B_tu_t + c_t,$$
where $x_t$ is the state, $u_t$ is the input, and $(A_t,B_t,c_t)$ may be random (but indpendent in time).

The goal is to minimize the average cost
$$J = \lim_{T\to\infty}\frac 1 T \sum_{t=0}^{T-1} g(x_t, u_t),$$
where $g$ is a convex stage cost. The stage cost can take on infinite values, to represent constraints on $(x_t, u_t)$.

We consider approximate dynamic programming (ADP) control policies of the form
$$\phi(x_t) = \text{argmin}_u \left(g(x_t, u) + \mathbf{E} \hat V(A_t x_t + B_t u + c_t)\right),$$
where $\hat V$ is a quadratic approximate value function of the form $\hat V(x) = (1/2)x^TPx + p^Tx$. If $\hat V$ is an optimal value function, then the ADP policy is also optimal.

## Example

In this example, we have a box-constrained linear quadratic regulator (LQR) problem, with dynamics
$$x_{t+1} = Ax_t + Bu_t + c_t,$$
where $A$ and $B$ are fixed and $c_t$ is a zero-mean Gaussian random variable. The stage cost is
$$g(x_t,u_t) = x_t^TQx_t + u_t^TR u_t + I(\|u_t\|_{\infty} \le u_{\max}),$$
where $Q$ and $R$ are positive semidefinite matrices and the last term is an indicator function that encodes the constraint that the entries of $u_t$ all have magnitude at most $u_{\max}$.

We can initialize a ```ControlProblem``` instance with state $x_t\in\mathbf{R}^{12}$ and input $u_t\in\mathbf{R}^{3}$ with
```python
n = 3
m = 2
problem = BoxLQRProblem.create_problem_instance(n, m, seed=0, processes=5)
```
Adding the extra argument ```processes=5``` lets us run simulations in parallel using 5 processes. By default, the cost is evaluated by simulating ```eval_trajectories=5``` trajectories, each for ```eval_horizon=2*10**4``` steps. 

We can get a quadratic lower bound on the optimal value function with
```python
V_lb = problem.V_lb()
```

To create an ADP policy and MPC policy with 30-step lookahead, we call
```python
policy = problem.create_policy(compile=True, name="box_lqr_policy", V=V_lb)
mpc = problem.create_policy(lookahead=30, compile=True, name="box_lqr_policy")
```
Setting the argument ```compile=True``` generates a custom solver implementation in C using [CVXPYgen](https://github.com/cvxgrp/cvxpygen).

To find an ADP policy using VGI, we run
```python
# initialize VGI method
vgi_method = vgi.VGI(
    problem,
    policy,
    vgi.QuadGradReg(),
    trajectory_len=50,
    num_trajectories=1,
    damping=0.5,
)
# find ADP policy by running VGI for 20 iterations
adp_policy = vgi_method(20)
```

To simulate the policy for 100 steps and plot the state trajectories, we can run
```python
simulation = problem.simulate(adp_policy, 100)

import matplotlib.pyplot as plt
plt.plot(simulation.states_matrix)
plt.show()
```

To evaluate the average cost of the policy via simulation, we can run
```python
adp_cost = problem.cost(adp_policy)
```

## Defining your own control problems

Examples of control problems can be found in [The examples folder](examples/). To set up a new control problem, we can inherit the ```ControlProblem``` class. For example, to create a linear quadratic regulator (LQR) problem, we might write
```python
from vgi import ControlProblem
class LQRProblem(ControlProblem):

    def __init__(self, A, B, Q, R):
        """Constructor for LQR problem"""
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        n, m = B.shape
        super().__init__(n, m)

    def step(self, x, u):
        """Dynamics for simulation. Returns next state, noise/observation/measurements, and stage cost"""
        c = np.random.randn(self.n)
        x_next = self.A @ x + self.B @ u + c
        stage_cost = x.T @ self.Q @ x + u.T @ self.R @ u
        return x_next, c, stage_cost

    def sample_initial_condition(self):
        return np.random.randn(self.n)
```
To create a corresponding policy for the ```LQRProblem```, we can create a ```LQRPolicy```, which inherits from ```COCP```, the class for convex optimization control policies (COCPs):
```python
import cvxpy as cp
from vgi import COCP

class LQRPolicy(COCP):
    def stage_cost(self, x, u):
        constraints = []
        return cp.quad_form(x, self.Q) + cp.quad_form(u, self.R), constraints
```
The stage cost function takes in CVXPY variables ```x``` and ```u```, and returns an expression for the stage cost, and any constraints on ```x``` and ```u```. The COCP constructor takes the state and control dimensions ```n``` and ```m``` as arguments, as well as any additional named parameters, such as the positive semidefinite cost matrices ```Q``` and ```R```, as well as the dynamics matrices ```A``` and ```B```.

For example, suppose we have an LQR problem with state dimension 3, input dimension 2, and randomly generated dynamics:
```python
# problem dimensions
import numpy as np
n = 3
m = 2

# generate random dynamics matrices
np.random.seed(0)
A = np.random.randn(n, n)
A /= np.max(np.abs(np.linalg.eigvals(A)))
B = np.random.randn(n, m)

# mean of c
c = np.zeros(n)

# cost parameters
Q = np.eye(n)
R = np.eye(m)

control_problem = LQRProblem(A, B, Q, R)
``` 
To create an ADP policy with randomly generated quadratic approximate value function,
```python
from vgi import QuadForm
V_hat = QuadForm.random(n)
adp_policy = LQRPolicy(n, m, Q=Q, R=R, A=A, B=B, c=c, V=V_hat)
```
To compile the policy to a custom solver implementation in C using CVXPYgen, add the argument ```compile=True``` as well as a directory name for the generated code, e.g. ```name="lqr_policy"```.

To simulate the policy for ```T``` steps, run
```python
T = 100
sim = control_problem.simulate(adp_policy, T, seed=0)
```
This yields a ```Simulation``` object. Calling ```sim.states_matrix``` gives a ```(T, n)``` matrix of the visited states.

To evaluate the average cost of the policy via simulation, we can run
```python
adp_cost = control_problem.cost(adp_policy, seed=0)
```
This runs ```eval_trajectories``` simulations starting from different randomly sampled initial conditions, each for ```eval_horizon``` steps, and returns the average cost. The simulations may optionally be run in parallel.

Those parameters may be set explicitly in the constructor for the control problem. For example, if we construct the ```LQRProblem``` as
```python
control_problem = LQRProblem(A, B, Q, R, eval_horizon=1000, eval_trajectories=5, processes=5)
```
then running
```python
adp_cost = control_problem.cost(adp_policy, seed=0)
```
will run 5 simulations in parallel, each for 1000 steps, and return the average cost on those trajectories.
