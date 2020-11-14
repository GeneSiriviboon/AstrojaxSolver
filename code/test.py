from ConstrainSolver import *
from astrojax import *
from colorama import Fore, Style
from colorama import init
init()
import numpy as np

def printResult(result):
    if result:
        print(f'{Fore.GREEN}Passed{Style.RESET_ALL}')
    else:
        print(f'{Fore.RED}Failed{Style.RESET_ALL}')



def ldotdot(pos, vel, acceleration, A, A_v, A_a, tension):
    rel1 = pos[:3] - A
    rel2 = pos[3:6] - pos[:3]

    l1 = np.linalg.norm(rel1) 
    l2 = np.linalg.norm(rel2)

    v_rel_1 = vel[:3] - A_v
    v_rel_2 = vel[3:6] - vel[:3]

    a1 = acceleration[:3]
    a2 = acceleration[3:6]

    dL = np.dot(a1 - A_a, rel1)/l1 \
        + np.linalg.norm(v_rel_1)**2/l1 \
        - np.dot(v_rel_1, rel1)**2 / l1**3\
        + np.dot(a2 - a1, rel2)/l2 \
        + np.linalg.norm(v_rel_2)**2/l2 \
        - np.dot(v_rel_2, rel2)**2 / l2**3
    
    return dL

def ldot(pos, vel, A, A_v):
    rel1 = pos[:3] - A
    rel2 = pos[3:] - pos[:3]

    l1 = np.linalg.norm(rel1) 
    l2 = np.linalg.norm(rel2)

    v_rel_1 = vel[:3] - A_v
    v_rel_2 = vel[3:] - vel[:3]
    
    return np.dot(v_rel_1, rel1)/l1 + np.dot(v_rel_2, rel2)/l2

def test1(num = 100):
    inputs = np.random.random([num, 22])
    poss = inputs[:, :6]
    vels = inputs[:, 6:12]
    endpoints = inputs[:, 12:15]
    endpoint_vels = inputs[:, 15:18]
    endpoint_accs = inputs[:, 18:21]
    accs = np.random.random([num, 6])

    model = ConstrainSolver()
    tensions = model(inputs)

    x1, y1 = constrainCalculator(poss, vels, accs, endpoints, endpoint_vels, endpoint_accs, tensions)
    
    x2, y2 = [], []
    for pos, vel, endpoint, endpoint_vel, endpoint_acc, acc, tension in zip(poss, vels, endpoints, endpoint_vels, endpoint_accs, accs, tensions):
        x2.append(ldot(pos, vel, endpoint, endpoint_vel))
        y2.append(ldotdot(pos, vel, acc, endpoint, endpoint_vel, endpoint_acc, tension))
    
    printResult(np.allclose(x1, x2)) 
    printResult(np.allclose(y1, y2)) 


def endpoint(t, deriv):

    f = np.array([0.0455, 0.0909, 0.1364]) * 8
    w = 2 * np.pi * f
    A_sin = np.array([[0.0152, 0.0097, 0.0040],[0, 0, 0] , [-0.0206, -0.0155, -0.0022]])
    A_cos = np.array([[-4.72e-4, 0.0075, -0.0043], [0, 0, 0] ,[-0.0076, 0.0072, -8.84e-06]])
    
    if deriv == 0:
        return np.sin(w * t)@A_sin + np.cos(w * t)@A_cos
    elif deriv == 1:
        return  (w * np.cos(w * t))@A_sin - (w * np.sin(w * t))@A_cos
    elif deriv == 2:
        return - (w**2 * np.sin(w * t))@A_sin - (w**2 * np.cos(w * t))@A_cos



def test2():

    t = np.linspace(0, 20 , num = 10000)

    initial_pos = np.hstack([endpoint(0, 0), endpoint(0, 0)]) + np.array([0.5, 0.0, 0.0, 0.5, 0.0, -0.5])
    initial_vel = np.hstack([endpoint(0, 1), endpoint(0, 1)])
    initial_state = np.hstack((initial_pos, initial_vel))
    l = length(initial_state, endpoint, 0)
    initial_state = initial_state/l

    # print('length:', length(initial_state, endpoint, 0))
    # print('ldot = {}'.format(ldot(initial_state[:6], initial_state[6:], endpoint(0,0), endpoint(0,1))))
    # print('adjusting initial state...')

    l_dot_init = lambda vel: ldot(initial_state[:6], vel, endpoint(0,0), endpoint(0,1))

    v_new = constrain(initial_vel, l_dot_init, tol = 1e-10)

    initial_state = np.hstack([initial_pos, v_new])
    # print('adjusted initial state')
    
    Dstate = dstate(m1 = 1, m2 = 1, t_end = t[-1], A = endpoint)
    Dstate_test = dstate(m1 = 1, m2 = 1, t_end = t[-1], A = endpoint, test = True)

    # print('checking initial condition...')
    assert(abs(ldot(initial_state[:6], initial_state[6:], endpoint(0,0), endpoint(0,1))) < 1e-10)
    # print('solving trajectory...')
    trajectory = odeint(Dstate, initial_state, t)
    trajectory_test = odeint(Dstate_test, initial_state, t)
    # print('trajectory obtained')

    printResult(np.allclose(trajectory, trajectory_test))
    


if __name__ == '__main__':
    test1()
    test2()

