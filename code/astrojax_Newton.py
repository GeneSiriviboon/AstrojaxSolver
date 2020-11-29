import numpy as np
from scipy.optimize import root, approx_fprime
from scipy.integrate import odeint

from util import *
from visualize import *



class Astrojax():

    def __init__(self):
        self.guess = np.array([0,0,-1,0,0,-1, 1])

    """
    state - representation

    state = [x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2]
    acceleration_tension = [a1x, a1y, a1z, a2x, a2y, a2z, T]
    m1 - mass of the upper ball
    m2 - mass of the lower ball
    A - function that takes in time t and derivative n and give out the nth derivative of endpoint over time
    """


    """
    creating constrain condition as a function of acceleration_tension

    [a - F/m, ldotdot]

    """
    def g_val(self, state, t):
        def g_func(acceleration_tension):


            rel1 = state[:3] 
            rel2 = state[3:6] - state[:3]

            l1 = np.linalg.norm(rel1) 
            l2 = np.linalg.norm(rel2)

            dL = l_dotdot(state[:6], state[6:12], acceleration_tension[:6], acceleration_tension[-1])


            g1 = acceleration_tension[:3] + acceleration_tension[6] * (rel1/l1 - rel2/l2) - np.array([0, 0, -1])
            g2 = acceleration_tension[3:6] + acceleration_tension[6] * (rel2/l2)  - np.array([0, 0, -1])
            return np.hstack([g1, g2, [dL]])
        
        return g_func

    """
    calculate the jacobian of the constrain
    """
    def jacobian(self, state, t):
        def jac(acceleration_tension):
            
            rel1 = state[:3] 
            rel2 = state[3:6] - state[:3]

            l1 = np.linalg.norm(rel1) 
            l2 = np.linalg.norm(rel2)

            r1 = np.hstack([rel1/l1 - rel2/l2, rel2/l2])
            r2 = np.hstack([rel1/l1 - rel2/l2, rel2/l2])
    
            j = np.eye(7)
            j[:-1, 6] = r2
            j[6, :-1] = r1
            j[6, 6] = 0
            
            return j
        
        return jac

    def solveAcc(self, state, t):
        Root = root(self.g_val(state = state, t = t), \
                self.guess, tol = 1e-10, jac = self.jacobian(state = state, t = t))

        return Root.x



    """
    solve newton equation to find acceleration
    """
    def dstate(self, t_end, p = True):
        def helper(state, t):
            acceleration_tension  = self.solveAcc(state, t)
        
            self.guess = acceleration_tension

            progress = t/t_end
            
            if p:
                print_progress(progress,length = 30)
        
            return np.hstack([state[6:], acceleration_tension[:-1]])

        return helper

    def trajectory(self, initial, t):
        Dstate = self.dstate(t_end = t[-1])
        print('checking initial condition...')
        assert(abs(ldot(initial[:6], initial[6:])) < 1e-10)
        print('solving trajectory...')
        trajectory = odeint(Dstate, initial, t)
        print()
        print('trajectory obtained')
        return trajectory



if __name__ == '__main__':
    

    
    astrojax = Astrojax()
    t = np.linspace(0, 40, num = 4000)
    
    initial_state = np.array([0.0, 0.0, -0.5, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    trajectory = astrojax.trajectory(initial_state, t)

    l = []
    l1 = []
    E = []
    for i in range(trajectory.shape[0]):
        l.append(length(trajectory[i]))
        E.append(Energy(trajectory[i]))
        
        endpoint = 0 #A * np.sin(w * t[i])
    
        rel1 = trajectory[i, :3] - endpoint
        rel2 = trajectory[i, 3:6] - trajectory[i, :3]

        l1.append(np.linalg.norm(rel1))

    l = np.array(l)
    l1 = np.array(l1)
    E = np.array(E)

    # plt.figure()
    # plt.xlabel('time', fontsize=15)
    # plt.ylabel('length', fontsize=15)
    
    # plt.plot(t, l, label='total length')
    # plt.plot(t, l1, label='length 1')
    # plt.plot(t, l - l1, label='length 2')
    
    # # plt.ylim([0, np.max(l) * 1.1])
    # plt.legend()
    # plt.savefig('BaselineLength.png', dpi = 300)
    
    
    plt.figure()
    plt.xlabel('time', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.plot(t, E, 'b')
    # plt.ylim([0, np.max(E) * 1.1])
    plt.savefig('BaselineEnergy.png', dpi = 300)
    plt.show()
    
    # print('rendering...')
    # animate2D(trajectory = trajectory[::7], time = t[::7], A = A, w = w, save = None)

    print(trajectory[:, 6:12])
