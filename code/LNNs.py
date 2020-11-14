import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
# from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial # reduces arguments to function by making some subset implicit

from functools import partial


def lagrangian(q, q_dot):
    m1 = 1; m2 = 1; g = 1;
    
    t1, t2, tr = q     # theta 1 and theta 2
    w1, w2, tr_dot = q_dot # omega 1 and omega 2
    
    l1 = jnp.sin(tr)**2 #(jnp.tanh(tr) + 1)/2
    l2 = 1 - l1

    l1_dot = jnp.sin(2 * tr) * tr_dot #r_dot/(2 * jnp.cosh(tr)**2) 
    l2_dot = -l1_dot 

    # kinetic energy (T)
    T1 = 0.5 * m1 * ((l1 * w1)**2 + l1_dot**2)
    T2 = 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 + l1_dot**2 + l2_dot**2 + \
                    2 * l1 * l2 * w1 * w2 * jnp.cos(t1 - t2) + \
                    2 * l1_dot * l2_dot * jnp.cos(t1 - t2))
    T = T1 + T2
    # potential energy (V)
    y1 = -l1 * jnp.cos(t1)
    y2 = y1 - l2 * jnp.cos(t2)
    V = m1 * g * y1 + m2 * g * y2
    return T - V

def equation_of_motion(lagrangian, state, t=None):
    q, q_t = jnp.split(state, 2)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
          @ (jax.grad(lagrangian, 0)(q, q_t)
             - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    dstate = jnp.concatenate([q_t, q_tt])
    print(dstate)
    return dstate

def solve_lagrangian(lagrangian, initial_state, **kwargs):
    # We currently run odeint on CPUs only, because its cost is dominated by
    # control flow, which is slow on GPUs.
    # @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(equation_of_motion, lagrangian),
                  initial_state, **kwargs)
    return f(initial_state)

# @partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times):
    return solve_lagrangian(lagrangian, initial_state, t=times, rtol=1e-10, atol=1e-10)

# choose an initial state
x0 = np.array([3*np.pi/7, 3*np.pi/4, np.pi/4, 0, 0, 0], dtype=np.float32)
noise = np.random.RandomState(0).randn(x0.size)
t = np.linspace(0, 10, num=200, dtype=np.float64)
# x_autograd = jax.device_get(solve_autograd(x0, t))
x_autograd = solve_autograd(x0, t)


for i in range(x_autograd.shape[0]):
    plt.clf()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    theta1 = x_autograd[i, 0]
    theta2 = x_autograd[i, 1]
    l1 = np.sin(x_autograd[i, 2])**2
    plt.plot([0, l1 * np.sin(theta1), l1 * np.sin(theta1) + (1- l1)* np.sin(theta2)], 
            [0, -l1 * np.cos(theta1), -l1 * np.cos(theta1) + - (1 - l1) * np.cos(theta2)], 'k')
    plt.pause(1e-3)

plt.show()