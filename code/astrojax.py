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
    def g_val(self, state, t, A, w):
        def g_func(acceleration_tension):

            endpoint = np.sin(w * t) * A

            rel1 = state[:3] - endpoint
            rel2 = state[3:6] - state[:3]

            l1 = np.linalg.norm(rel1) 
            l2 = np.linalg.norm(rel2)

            dL = l_dotdot(state[:6], state[6:12], acceleration_tension[:6],
                            A, w, t,
                            acceleration_tension[-1])


            g1 = acceleration_tension[:3] + acceleration_tension[6] * (rel1/l1 - rel2/l2) - np.array([0, 0, -1])
            g2 = acceleration_tension[3:6] + acceleration_tension[6] * (rel2/l2)  - np.array([0, 0, -1])
            return np.hstack([g1, g2, [dL]])
        
        return g_func

    """
    calculate the jacobian of the constrain
    """
    def jacobian(self, state, t, A, w):
        def jac(acceleration_tension):
            
            rel1 = state[:3] - A * np.sin(w * t)
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

    def solveAcc(self, state, t, A, w):
        Root = root(self.g_val(state = state, t = t, A = A, w = w), \
                self.guess, tol = 1e-10, jac = self.jacobian(state = state, t = t, A = A, w = w))

        return Root.x



    """
    solve newton equation to find acceleration
    """
    def dstate(self, t_end, A, w, p = True):
        def helper(state, t):
            acceleration_tension  = self.solveAcc(state, t, A, w)
        
            self.guess = acceleration_tension

            progress = t/t_end
            
            if p:
                print_progress(progress,length = 30)
        
            return np.hstack([state[6:], acceleration_tension[:-1]])

        return helper

    def trajectory(self, initial, t, A, w):
        Dstate = self.dstate(t_end = t[-1], A = A, w = w)
        print('checking initial condition...')
        assert(abs(ldot(initial[:6], initial[6:], A, w, 0)) < 1e-10)
        print('solving trajectory...')
        trajectory = odeint(Dstate, initial_state, t)
        print()
        print('trajectory obtained')
        return trajectory





if __name__ == '__main__':
    
    astrojax = Astrojax()
    t = np.linspace(0, 50 , num = 10000)
    A = np.zeros([1, 3])
    w = 2 * np.pi * np.ones(1)
    t0 = np.zeros([1])
    
    initial_state = generate_data2D(1, A, w, t0)[0, :12]
    initial_state = np.array([0.0, 0.0, -0.5, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    trajectory = astrojax.trajectory(initial_state, t, A[0], w[0])
    
    print('rendering...')
    animate2D(trajectory = trajectory[::7], time = t[::7], A = A, w = w, save = None)

    print(trajectory[:, 6:12])
