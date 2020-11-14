import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from util import *
from preprocess import *

tf.keras.backend.set_floatx('float64')
class ConstrainSolver (tf.keras.Model):

    def __init__(self):
        super(ConstrainSolver, self).__init__()

        self.batch_sz = 2048
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.dt = 1e-2

        #---------------------

        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Dropout(.2))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        # self.model.add(tf.keras.layers.Dropout(.2))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        # self.model.add(tf.keras.layers.Dropout(.2))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        # self.model.add(tf.keras.layers.Dropout(.2))
        self.model.add(tf.keras.layers.Dense(1))

    def call(self, inputs):
        pos = inputs[:, :3]
        vel = inputs[:, 3:6]
        l = tf.norm(pos, axis = 1)

        # print((self.model(inputs).shape))
        # print((tf.reduce_sum(vel**2, axis = 1) /l).shape)
        return self.model(inputs) * tf.reshape(tf.reduce_sum(vel**2, axis = 1) /l, [-1, 1])

    def loss(self, inputs, tension):

        a = tension + np.array([[0.0, 0.0, -1.0]])

        pos = inputs[:, :3]
        vel = inputs[:, 3:6]
        l = tf.norm(pos, axis = 1)
        l_dot_dot = dot(a, pos)/l + tf.reduce_sum(vel**2, axis = 1)/l - dot(vel, pos)**2/l**3

        # print(l_dot_dot.shape)
        T_labels = tf.reduce_sum(vel**2, axis = 1)/l - dot(vel, pos)**2/l**3 - pos[:, 2]/ l

        # print('xxxxxxx', T_labels)
        # print('yyyyyyy', tension)

        return tf.reduce_mean(tf.keras.losses.mean_squared_error(T_labels, tension))#tf.reduce_mean(l_dot_dot**2)

def train(model, inputs):
    sinputs = tf.random.shuffle(inputs)
    loss_sum = 0
    for i in range(0, inputs.shape[0], model.batch_sz):    
        with tf.GradientTape() as tape:
            tension = model(sinputs[i: i+model.batch_sz])
            loss = model.loss(sinputs[i: i+model.batch_sz], tension)   
            loss_sum+=loss     
        gradients = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss_sum /(inputs.shape[0] // model.batch_sz)

def test(model, inputs):
    loss_sum = 0
    for i in range(0, inputs.shape[0], model.batch_sz):    
        tension = model(inputs[i: i+model.batch_sz])
        loss = model.loss(inputs[i: i+model.batch_sz], tension)   
        loss_sum += loss
    
    return loss_sum /(inputs.shape[0] // model.batch_sz)


def main():
    model = ConstrainSolver()
    num_train = 50000
    num_test = 10000
    num_pre_epoch = 1000
    num_epoch = 1000

    theta = np.random.random([num_train]) * np.pi * 2
    phi = np.random.random([num_train]) * np.pi

    theta_dot = np.random.random([num_train]) * np.pi * 2 * 3
    phi_dot = np.random.random([num_train]) * np.pi * 3
   
    pos = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis = 1)
    vel = np.stack([np.cos(theta) * np.cos(phi) * theta_dot - np.sin(theta) * np.sin(phi) * phi_dot,
                     np.cos(theta) * np.sin(phi) * theta_dot + np.sin(theta) * np.cos(phi) * phi_dot, 
                     -np.sin(theta) * theta_dot], axis = 1)

    train_inputs = np.hstack([pos, vel])
    print(model(train_inputs[:model.batch_sz]))
    print(train_inputs.shape)

    for i in range(num_epoch):
        train_loss = train(model, train_inputs)
        # test_loss = test(model, test_inputs)
        print('Epoch: {:d} train: {:f}'.format(i+1, train_loss))
        plt.plot(i, train_loss, 'b.', label = 'train loss')
        # plt.plot(i, test_loss, 'g.', label = 'test loss')
    
        plt.pause(0.01)

        if i == 0 :
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.yscale('log')

            
    

    plt.show()


if __name__ == '__main__':
    main()


    
        
        



