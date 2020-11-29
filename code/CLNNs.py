import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
# from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial # reduces arguments to function by making some subset implicit

from scipy.optimize.nonlin import Jacobian

from jax.config import config; 
config.update("jax_enable_x64", True)

def equation_of_motion(f, dPhi, dPhi_dot, state, t=None):
    q, q_t = jnp.split(state, 2)
    dphi = dPhi(q,0).reshape([1, -1])
    dphi_dot = dPhi_dot(q, q_t, 0).reshape([1, -1])
    q_tt = f - dphi.T @ jnp.linalg.pinv(dphi @ dphi.T) @(dphi @ f + dphi_dot @ q_t)
    dstate = jnp.concatenate([q_t, q_tt])
    return dstate


def solve_lagrangian(force, dPhi, dPhi_dot, initial_state, **kwargs):
    # We currently run odeint on CPUs only, because its cost is dominated by
    # control flow, which is slow on GPUs.
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(equation_of_motion, force, dPhi, dPhi_dot),
                  initial_state, **kwargs)
    return f(initial_state)

def solve_autograd(f, dPhi, dPhi_t):
    @partial(jax.jit, backend='cpu')
    def helper(initial_state, times):
        return solve_lagrangian(f, dPhi, dPhi_t, initial_state, t=times, rtol=1e-10, atol=1e-10)
    return helper



if __name__ == '__main__':
    
    state = np.array([1.0, 0.0, 0.0, 0.0])
    states = np.random.random([100, 4])
    f = np.array([0,-1])
    times = np.linspace(0, 10, num = 100)
    dPhi = lambda q, t: 2 * q
    dPhi_t = lambda q, q_t, t: q_t @ jax.jacobian(dPhi)(q, t).T
    print(equation_of_motion(f, dPhi, dPhi_t, state))
    ans = [equation_of_motion(f, dPhi, dPhi_t, state) for state in states ]
    maps_ans = jax.vmap(partial(equation_of_motion, f, dPhi, dPhi_t), 0)(states)

    print(np.allclose(ans, maps_ans))



