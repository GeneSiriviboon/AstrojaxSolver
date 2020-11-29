import jax.numpy as jnp
from matplotlib.pyplot import plot
import numpy as np
from astrojax import *
from functools import partial
from util import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualize import *


def randomPolar(num):
    theta1s = np.random.random(num) * np.pi
    theta2s = np.random.random(num) * np.pi
    phi1s = np.random.random(num) * np.pi * 2 
    phi2s = np.random.random(num) * np.pi * 2 
    theta1sdot = np.random.random(num) * np.pi
    theta2sdot = np.random.random(num) * np.pi
    phi1sdot = np.random.random(num) * np.pi * 2
    phi2sdot = np.random.random(num) * np.pi * 2
    r = np.random.random(num)
    r_dot = np.random.random(num)

    return (theta1s,phi1s,theta2s,phi2s, r), (theta1sdot,phi1sdot,theta2sdot,phi2sdot, r_dot)


def randomStateInitializer(num):

    x, v = randomPolar(num)

    return (x, v)

def randomPair(num, epsilon):
    x, v = randomPolar(num) 
    state1 =  polar2Cartesian(x, v)

    theta1, phi1, theta2, phi2, r = x
    x2 = (theta1, phi1, theta2 + epsilon, phi2, r)
    
    # state2 = polar2Cartesian((theta1, phi1, theta2 + epsilon, phi2, r), v)

    return (x, v), (x2, v)

def solveAstrojaxs(states, t):
    return jax.vmap(partial(solveAstrojax, t = t)) (states)

def lyupunov(PolarStates1, PolarStates2, t = 3, num = 400):
    time = np.linspace(0, t, num = num)

    states1 = polar2Cartesian(PolarStates1[0], PolarStates1[1])
    states2 = polar2Cartesian(PolarStates2[0], PolarStates2[1])
    
    traj1 = np.array([solveAstrojax(state1, time) for state1 in tqdm(states1)])
    traj2 = np.array([solveAstrojax(state2, time) for state2 in tqdm(states2)])

    diffs = np.sum((traj1 - traj2)**2, axis = -1)**0.5
    slope = [np.polyfit(time, np.log(diff), 1)[0] for diff in diffs]
    
    return slope, traj1, traj2

def Cartesian2Polar(traj, t):
    r = np.sum(traj[:, :3]**2, axis = 1)**0.5

    theta1 = np.unwrap(np.arccos(traj[:, 2]/r) * np.sign(traj[:, 0]))
    theta2 = np.unwrap(np.arccos((traj[:, 5] - traj[:, 2])/(1-r)) * np.sign(traj[:, 3] - traj[:, 0]))
    # theta1dot = - traj[:, 8] /(r * np.sin(theta1))
    # theta2dot = - (traj[:, 11] - traj[:, 8]) /((1-r) * np.sin(theta2))
    theta1dot = (theta1[2:] - theta1[:-2])/(t[1] - t[0])/2
    theta2dot = (theta2[2:] - theta2[:-2])/(t[1] - t[0])/2
    r_dot = (r[2:] - r[:-2])/(t[1] - t[0])/2
    return theta1[1:-1], theta2[1:-1], theta1dot, theta2dot, r[1:-1], r_dot, t[1:-1] 


if __name__ == '__main__':
    theta1s = np.array([np.pi]); theta2s = np.array([np.pi - 1e-3])
    phi1s = np.array([0]); phi2s = np.array([0])
    theta1sdot = np.array([0]) ; theta2sdot = np.array([0]); phi1sdot = np.array([0]); phi2sdot = np.array([0])

    r = np.array([0.5])
    r_dot = np.array([0])

    x, v = (theta1s,phi1s,theta2s,phi2s, r), (theta1sdot,phi1sdot,theta2sdot,phi2sdot, r_dot)

    t = np.linspace(0, 20000, num = 2000000)

    state1 = polar2Cartesian(x, v)[0]

    traj = solveAstrojax(state1, t)

    theta1, theta2, theta1dot, theta2dot, r, r_dot, t1 = Cartesian2Polar(traj, t)
    theta1 = theta1 % (2 * np.pi) - np.pi
    theta2 = theta2 % (2 * np.pi) - np.pi
    # plt.plot(t1, theta1dot, 'r.')
    # plt.plot(t1, theta1, 'b.')
    zero_crossings = np.where(np.diff(np.sign(theta1)))[0]
    zero_crossings = zero_crossings[np.abs(theta1[zero_crossings]) < 1]
    zero_crossings = zero_crossings[theta1dot[zero_crossings] > 0]
    zero_crossings = zero_crossings[r[zero_crossings] > 1e-1]

    fig, axs = plt.subplots(1, 2, figsize = [8,4])
    for i in range(zero_crossings.shape[0]):
        axs[0].plot(theta2[zero_crossings[i]], theta2dot[zero_crossings[i]], 'k.', markersize = 3)
        axs[0].set_xlabel(r'$\theta_2$', fontsize=20)
        axs[0].set_ylabel(r'$\dot{\theta}_2$', fontsize=20)
        axs[0].set_ylim([-20, 20])
        axs[1].plot(r[zero_crossings[i]], r_dot[zero_crossings[i]], 'k.', markersize = 3)
        axs[1].set_xlabel(r'$r$', fontsize=20)
        axs[1].set_ylabel(r'$\dot{r}$', fontsize=20)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(r'The Poincare Section for $\theta_1$ = {:.3f} $\theta_2$ = {:.3f}'.format(theta1s[0], theta2s[0]), fontsize = 15)
        plt.pause(0.01)
    plt.show()
# plt.savefig('low_energy1.png', dpi = 300)

    # t = np.linspace(0, 40, num = 4000)
    # initial_state = np.array([0.0, 0.0, -0.5, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # trajectory = solveAstrojax(initial_state, t)

    # l = []
    # l1 = []
    # E = []
    # for i in range(trajectory.shape[0]):
    #     l.append(length(trajectory[i]))
    #     E.append(Energy(trajectory[i]))
        
    #     endpoint = 0 #A * np.sin(w * t[i])
    
    #     rel1 = trajectory[i, :3] - endpoint
    #     rel2 = trajectory[i, 3:6] - trajectory[i, :3]

    #     l1.append(np.linalg.norm(rel1))

    # l = np.array(l)
    # l1 = np.array(l1)
    # E = np.array(E)

    # plt.figure()
    # plt.xlabel('time', fontsize=15)
    # plt.ylabel('length', fontsize=15)
    
    # plt.plot(t, l, label='total length')
    # plt.plot(t, l1, label='length 1')
    # plt.plot(t, l - l1, label='length 2')
    
    # plt.ylim([0, np.max(l) * 1.1])
    # plt.legend()
    # plt.savefig('BaselineLengthAutoGrad.png', dpi = 300)
    
    
    # plt.figure()
    # plt.xlabel('time', fontsize=15)
    # plt.ylabel('Energy', fontsize=15)
    # plt.plot(t, E, 'b')
    # # plt.ylim([0, np.max(E) * 1.1])
    # plt.savefig('BaselineEnergyAutoGrad.png', dpi = 300)
    # plt.show()
    
    # # print('rendering...')
    # # animate2D(trajectory = trajectory[::7], time = t[::7], A = A, w = w, save = None)

