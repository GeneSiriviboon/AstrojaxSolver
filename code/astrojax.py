from CLNNs import *
import jax.numpy as jnp
import numpy as np
from util import *

from jax.config import config; 
config.update("jax_enable_x64", True)

def DPhi(q, t):
    q1, q2 = jnp.split(q, 2)
    
    rel1 = q1
    rel2 = q2 - q1

    l1 = jnp.linalg.norm(rel1) 
    l2 = jnp.linalg.norm(rel2)

    r1 = jnp.hstack([rel1/l1 - rel2/l2, rel2/l2])
    return r1

def DPhi_dot(q, q_dot,  t):
    return q_dot @ jax.jacobian(DPhi)(q, t).T

def solveAstrojax(initial, t):
    f = np.array([0, 0, -1, 0, 0, -1])
    trajectory = jax.device_get(solve_autograd(f, DPhi, DPhi_dot)(initial, t))
    return trajectory


if __name__ == '__main__':
    state1 = np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0,  
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    t = np.linspace(0, 40, num = 4000)
    trajectory1 = solveAstrojax(state1, t)

    E = [Energy(state) for state in trajectory1]
    l = [length(state) for state in trajectory1]

    plt.plot(t, E, 'g')
    plt.plot(t, l, 'r')
    plt.show()
    # for i in range(0, trajectory.shape[0], 10):
    #     plt.clf()
    #     plt.xlim([-2, 2])
    #     plt.ylim([-2, 2])
    #     plt.axes().set_aspect('equal')
        
    #     plt.plot([0, trajectory[i, 0], trajectory[i, 2]], 
    #             [0, trajectory[i, 1], trajectory[i, 3]], 'k')
    #     plt.pause(1e-4)
        

    # plt.show()


