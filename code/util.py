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
def length(state):
    q1, q2, q1_t, q2_t = np.split(state, 4)
    
    rel1 = q1
    rel2 = q2 - q1

    l1 = np.linalg.norm(rel1) 
    l2 = np.linalg.norm(rel2)
    return l1+l2

def Energy(state):
    q1, q2, q1_t, q2_t = np.split(state, 4)
    return 0.5*np.sum(q1_t**2 + q2_t **2) + q1[-1] + q2[-1] + 2

"""
calculate rate of rope length changes (use to check the constrain)
"""
def ldot(pos, vel):

   
    rel1 = pos[:3] 
    rel2 = pos[3:] - pos[:3]

    l1 = np.linalg.norm(rel1) 
    l2 = np.linalg.norm(rel2)

    v_rel_1 = vel[:3] 
    v_rel_2 = vel[3:] - vel[:3]
    
    return np.dot(v_rel_1, rel1)/l1 + np.dot(v_rel_2, rel2)/l2

# numpy version
def l_dotdot(pos, vel, acc, tension):

    rel1 = pos[:3] 
    rel2 = pos[3:6] - pos[:3]

    l1 = np.linalg.norm(rel1) 
    l2 = np.linalg.norm(rel2)

    v_rel_1 = vel[:3]
    v_rel_2 = vel[3:6] - vel[:3]

    dL = np.dot(acc[:3], rel1)/l1 \
        + np.linalg.norm(v_rel_1)**2/l1 \
        - np.dot(v_rel_1, rel1)**2 / l1**3\
        + np.dot(acc[3:6] - acc[:3], rel2)/l2 \
        + np.linalg.norm(v_rel_2)**2/l2 \
        - np.dot(v_rel_2, rel2)**2 / l2**3
    
    return dL

def polar2Cartesian(polarPos, polarVel):
    (theta1, phi1, theta2, phi2, r) = polarPos
    (theta1dot, phi1dot, theta2dot, phi2dot, rdot) = polarVel

    sinTheta1 = np.sin(theta1)
    cosTheta1 = np.cos(theta1)
    sinTheta2 = np.sin(theta2)
    cosTheta2 = np.cos(theta2)
    sinPhi1 = np.sin(phi1)
    cosPhi1 = np.cos(phi1)
    sinPhi2 = np.sin(phi2)
    cosPhi2 = np.cos(phi2)

    l1 = r[:, np.newaxis]
    l1_dot = rdot[:, np.newaxis]

    Omega1 = 2 * np.pi * np.stack([-theta1dot * sinPhi1, theta1dot * cosPhi1, phi1dot], axis = 1)
    Omega2 = 2 * np.pi * np.stack([-theta2dot * sinPhi2, theta2dot * cosPhi2, phi2dot], axis = 1)
  
    s1 = np.stack([sinTheta1 * cosPhi1, sinTheta1 * sinPhi1, cosTheta1], axis = 1)
    s2 = np.stack([sinTheta2 * cosPhi2, sinTheta2 * sinPhi2, cosTheta2], axis = 1)

    pos = np.hstack([s1 * l1, s1 * l1 + s2 * (1 - l1)]) 

    vel1 = np.hstack([s1 * l1_dot, s1 * l1_dot - s2 * l1_dot])
    vel2 = np.hstack([np.cross(Omega1, s1 * l1), 
                           np.cross(Omega1, s1 * l1) + np.cross(Omega2, s2 * (1-l1))])

    vel = vel1 + vel2 

    return np.hstack([pos, vel])
