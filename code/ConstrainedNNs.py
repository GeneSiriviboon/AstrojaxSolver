import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from util import *
from preprocess import *
import os

path = os.getcwd()

tf.keras.backend.set_floatx('float64')
class ConstrainNNs (tf.keras.Model):

    def __init__(self):
        super(ConstrainNNs, self).__init__()

        self.batch_sz = 1024
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        #---------------------

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(5))

    def call(self, inputs, acc):
        concat_input = np.hstack([inputs, acc])
        return self.model(concat_input) 


    def loss(self, states, endpoint_acc, output):

        polar1 = states[:, :2]
        polar2 = states[:, 2:4]
        r = tf.reshape(states[:, 4], [-1, 1])

        theta_dot1 = states[:, 5:7]
        theta_dot2 = states[:, 7:9]
        r_dot = tf.reshape(states[:, 9], [-1, 1])

        theta_dot_dot1 = output[:, :2]
        theta_dot_dot2 = output[:, 2:4]
        r_dot_dot = tf.reshape(output[:, 4], [-1, 1])


        r1 = tf.stack([r[:, 0] * tf.math.sin(polar1[:, 0]) * tf.math.cos(polar1[:, 1]),
                       r[:, 0] * tf.math.sin(polar1[:, 0]) * tf.math.sin(polar1[:, 1]),
                       r[:, 0] * tf.math.cos(polar1[:, 0])], axis = 1)

        r2 = tf.stack([(1-r)[:, 0] * tf.math.sin(polar2[:, 0]) * tf.math.cos(polar2[:, 1]),
                       (1-r)[:, 0] * tf.math.sin(polar2[:, 0]) * tf.math.sin(polar2[:, 1]),
                       (1-r)[:, 0] * tf.math.cos(polar2[:, 0])], axis = 1)

        r1_hat, _ = tf.linalg.normalize(r1, axis = 1) 
        r2_hat, _ = tf.linalg.normalize(r2, axis = 1)
        
        omega1 = tf.stack([-theta_dot1[:, 0] * tf.math.sin(polar1[:, 1]),
                          -theta_dot1[:, 0] * tf.math.cos(polar1[:, 1]),
                          -theta_dot1[:, 1]], axis = 1)
        
        omega2 = tf.stack([-theta_dot2[:, 0] * tf.math.sin(polar2[:, 1]),
                          -theta_dot2[:, 0] * tf.math.cos(polar2[:, 1]),
                          -theta_dot2[:, 1]], axis = 1)

        omega_dot1 =  tf.stack([-theta_dot_dot1[:, 0] * tf.math.sin(polar1[:, 1]) - \
                                            theta_dot1[:,0] * tf.math.cos(polar1[:,1]) * theta_dot_dot1[:, 1],
                          theta_dot_dot1[:, 0] * tf.math.cos(polar1[:, 1]) - \
                                            theta_dot1[:,0] * tf.math.sin(polar1[:,1]) * theta_dot_dot1[:, 1],
                          -theta_dot_dot1[:, 1]], axis = 1)

        omega_dot2 =  tf.stack([-theta_dot_dot2[:, 0] * tf.math.sin(polar2[:, 1]) - \
                                            theta_dot2[:,0] * tf.math.cos(polar2[:,1]) * theta_dot_dot2[:, 1],
                          theta_dot_dot2[:, 0] * tf.math.cos(polar2[:, 1]) - \
                                            theta_dot2[:,0] * tf.math.sin(polar2[:,1]) * theta_dot_dot2[:, 1],
                          -theta_dot_dot2[:, 1]], axis = 1)

        omega_r_1 = tf.linalg.cross(omega1, r1_hat)
        omega_r_2 = tf.linalg.cross(omega2, r2_hat)

        r1_dot_dot = r_dot_dot * r1_hat + \
                        2 * r_dot * omega_r_1 + \
                        r * tf.linalg.cross(omega_dot1, r1_hat) + \
                        r * tf.linalg.cross(omega1, omega_r_1)
        
        r2_dot_dot = r1_dot_dot - r_dot_dot * r2_hat + \
                        - 2 * r_dot * omega_r_2 + \
                        (1 - r) * tf.linalg.cross(omega_dot2, r2_hat) + \
                        (1 - r) * tf.linalg.cross(omega2, omega_r_2)

        z = np.array([[0, 0, 1]])
        
        x1 = r1_dot_dot +  endpoint_acc + z 
        x2 = r2_dot_dot +  endpoint_acc + z + x1 

        G1 = tf.norm(x1 + tf.reshape(tf.norm(x1, axis = 1), [-1, 1]) * r2_hat, axis = 1)
        G2 = tf.norm(x1 + x2 + tf.reshape(tf.norm(x1 + x2, axis = 1), [-1, 1]) * r1_hat, axis = 1)
        G3 = dot(x1 + x2, r1_hat) - dot(x2, r2_hat)

        return tf.reduce_mean(G1**2 + G2**2 + G3**2)

def train(model, inputs, acc):
    idx = tf.random.shuffle(np.arange(inputs.shape[0]))
    sinputs = tf.gather(inputs, idx)
    sacc = tf.gather(acc, idx)
    loss_sum = 0
    for i in range(0, inputs.shape[0], model.batch_sz):    
        with tf.GradientTape() as tape:
            out = model(sinputs[i: i+model.batch_sz], sacc[i: i+model.batch_sz])
            loss = model.loss(sinputs[i: i+model.batch_sz], sacc[i: i+model.batch_sz], out)   
            loss_sum+=loss     
        gradients = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))


    return loss_sum /(inputs.shape[0] // model.batch_sz)



def test(model, inputs, acc):
    loss_sum = 0
    for i in range(0, inputs.shape[0], model.batch_sz):    
        out = model(inputs[i: i+model.batch_sz], acc[i: i+model.batch_sz])
        loss = model.loss(inputs[i: i+model.batch_sz], acc[i: i+model.batch_sz], out) 
        loss_sum += loss
    
    return loss_sum /(inputs.shape[0] // model.batch_sz)




def main():
    model = ConstrainNNs()
    num_train = 50000

    acc = np.zeros([num_train, 3])

    states = np.random.random([num_train, 10]) * 2 - 1
    states[:, :4] *= np.pi * 2
    states[:, 4] = (states[:, 4] + 1)/2
    # states[:, 5:] *= 2

    print(test(model, states, acc), end = '\r')
    for i in range(1000):
        if i % 10 == 0:
            print()
            print('randomize data...')
            states = np.random.random([num_train, 10]) * 2 - 1
            states[:, :4] *= np.pi * 2
            states[:, 5:] *= 2
        
        loss = train(model, states, acc)
        print(loss, end = '\r')
        plt.plot(i, loss, '.k')
        plt.pause(0.01)

    print()
    model.save_weights(path + '/checkpoint')

        

    

if __name__ == '__main__':
    main()


    
        
        



