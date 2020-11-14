import tensorflow as tf
import numpy as np


tf.keras.backend.set_floatx('float64')

"""
print the progress of solving trajectory
 - input
    progress - float from 0(start) - 1(finish)'
    length - langth of the progressbar
"""
def print_progress(progress,length = 30):
    n = int(progress * length)
    print('|',*['#']*n, *[' ']*(length - n), '|', ' {:d}'.format(int(progress * 100)), '%', end = '\r', sep = '')



def dot(u, v):
    return tf.einsum('ij,ij->i',u, v)

"""
calculate the length of the rope (use to check the constrain)
"""
def length(state, A, w, t):
    endpoint = A * np.sin(w * t)
    
    rel1 = state[:3] - endpoint
    rel2 = state[3:6] - state[:3]

    l1 = np.linalg.norm(rel1) 
    l2 = np.linalg.norm(rel2)
    return l1+l2


def Energy(state):
    return 0.5*np.sum(state[6:12]**2) + state[2] + state[5] + 2

"""
calculate rate of rope length changes (use to check the constrain)
"""
def ldot(pos, vel, A, w, t):

    endpoint = A * np.sin(w * t)
    endpointVel = (A * w) * np.cos(w * t)

    rel1 = pos[:3] - endpoint
    rel2 = pos[3:] - pos[:3]

    l1 = np.linalg.norm(rel1) 
    l2 = np.linalg.norm(rel2)

    v_rel_1 = vel[:3] - endpointVel
    v_rel_2 = vel[3:] - vel[:3]
    
    return np.dot(v_rel_1, rel1)/l1 + np.dot(v_rel_2, rel2)/l2

# numpy version
def l_dotdot(pos, vel, acc, A, w, t, tension):

    endpoint = A * np.sin(w * t)
    endpointVel = (A * w) * np.cos(w * t)
    endpointAcc =  - A * w**2 * np.sin(w * t)

    rel1 = pos[:3] - endpoint
    rel2 = pos[3:6] - pos[:3]

    l1 = np.linalg.norm(rel1) 
    l2 = np.linalg.norm(rel2)

    v_rel_1 = vel[:3] - endpointVel
    v_rel_2 = vel[3:6] - vel[:3]

    dL = np.dot(acc[:3] - endpointAcc, rel1)/l1 \
        + np.linalg.norm(v_rel_1)**2/l1 \
        - np.dot(v_rel_1, rel1)**2 / l1**3\
        + np.dot(acc[3:6] - acc[:3], rel2)/l2 \
        + np.linalg.norm(v_rel_2)**2/l2 \
        - np.dot(v_rel_2, rel2)**2 / l2**3
    
    return dL

# Tensorflow Paralellize version
def constrainCalculator(pos, vel, acc, endpoint, endpointVel, endpointAcc, tension):

    rel1 = pos[:, :3] - endpoint
    rel2 = pos[:, 3:6] - pos[:, :3]

    l1 = tf.norm(rel1, axis = 1)
    l2 = tf.norm(rel2, axis = 1)

    v_rel_1 = vel[:, :3] - endpointVel
    v_rel_2 = vel[:, 3:] - vel[:, :3]

    rel1_norm, _ = tf.linalg.normalize(rel1, axis = 1)
    rel2_norm, _ = tf.linalg.normalize(rel2, axis = 1)
    
    g = np.array([[0.0, 0.0, -1.0]])

    a1 = acc[:,:3] if acc is not None else g - tension * (rel1_norm - rel2_norm)
    a2 = acc[:,3:6] if acc is not None else g - tension * rel2_norm 
    
    l_dot = dot(v_rel_1, rel1)/l1 + dot(v_rel_2, rel2)/l2

    l_dotdot = dot(a1 - endpointAcc, rel1)/l1 \
            + tf.norm(v_rel_1, axis = 1)**2/l1 \
            - dot(v_rel_1, rel1)**2 / l1**3 \
            + dot(a2 - a1, rel2)/l2 \
            + tf.norm(v_rel_2, axis = 1)**2/l2 \
            - dot(v_rel_2, rel2)**2 / l2**3

    return l_dot, l_dotdot

def polar2Cartesian(polarPos, polarVel, A, w, t):
    sinTheta1 = np.sin(2 * np.pi * polarPos[:,0])
    cosTheta1 = np.cos(2 * np.pi * polarPos[:,0])
    sinTheta2 = np.sin(2 * np.pi * polarPos[:,2])
    cosTheta2 = np.cos(2 * np.pi * polarPos[:,2])
    sinPhi1 = np.sin(2 * np.pi * polarPos[:,1])
    cosPhi1 = np.cos(2 * np.pi * polarPos[:,1])
    sinPhi2 = np.sin(2 * np.pi * polarPos[:,3])
    cosPhi2 = np.cos(2 * np.pi * polarPos[:,3])
   
    endpoint = A * np.sin(w * t)[:, np.newaxis] 
    endpointVel = A * (w * np.cos(w * t))[:, np.newaxis] 
    endpointAcc =  - A * (w**2 * np.sin(w * t))[:, np.newaxis] 

    l1 = polarPos[:, 4, np.newaxis]
    l1_dot = polarVel[:, 4, np.newaxis]

    Omega1 = 2 * np.pi * np.stack([-polarVel[:,0] * sinPhi1, polarVel[:,0] * cosPhi1, polarVel[:,1]], axis = 1)
    Omega2 = 2 * np.pi * np.stack([-polarVel[:,2] * sinPhi2, polarVel[:,2] * cosPhi2, polarVel[:,3]], axis = 1)
  
    s1 = np.stack([sinTheta1 * cosPhi1, sinTheta1 * sinPhi1, cosTheta1], axis = 1)
    s2 = np.stack([sinTheta2 * cosPhi2, sinTheta2 * sinPhi2, cosTheta2], axis = 1)

    pos = np.hstack([s1 * l1, s1 * l1 + s2 * (1 - l1)]) + np.tile(endpoint, 2)

    vel1 = np.hstack([s1 * l1_dot, s1 * l1_dot - s2 * l1_dot])
    vel2 = np.hstack([np.cross(Omega1, s1 * l1), 
                           np.cross(Omega1, s1 * l1) + np.cross(Omega2, s2 * (1-l1))])

    vel = vel1 + vel2 + np.tile(endpointVel, 2)

    return np.hstack([pos, vel, endpoint, endpointVel, endpointAcc])



def generate_data(num_data, A = None, w = None, t = None):
    polars = np.random.random([num_data, 10])
    if A is None:
        A = np.random.random([num_data, 3])  
        w = np.random.random([num_data]) * 3
        t = np.random.random([num_data]) * 20
    
    inputs = polar2Cartesian(polars[:,:5], polars[:, 5:10], A, w, t)
    print('generated dataset')
    return inputs

def generate_data2D(num_data, A = None, w = None, t = None):
    polars = np.random.random([num_data, 10])
    if A is None:
        A = np.random.random([num_data, 3])  
        A[:, 1] = 0
        w = np.random.random([num_data]) * 3
        t = np.random.random([num_data]) * 20

    polars[:, 1] = polars[:, 3] = polars[:, 5] = polars[:, 7] = 0
    inputs = polar2Cartesian(polars[:,:5], polars[:, 5:10] * 3, A, w, t)
    print('generated dataset')
    return inputs


def constrain(initial_vel, condition, tol = 1e-10, lamda = 0.5):
    while(abs(condition(initial_vel)) > tol):
        df = approx_fprime(initial_vel, condition, tol)
        initial_vel = initial_vel - df/np.linalg.norm(df) * condition(initial_vel) * lamda

    return initial_vel










