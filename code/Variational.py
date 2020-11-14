import tensorflow as tf
import numpy as np

def L(x):
    v = (x[:-1] - x[1:])/(t[1] - t[0])
    return v**2 - x[:-1]**2

def S(xs, vs, ts):
    return tf.reduce_sum(L(xs, vs, ts))

