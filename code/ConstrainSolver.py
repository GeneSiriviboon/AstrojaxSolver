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

        self.batch_sz = 1024
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.dt = 1e-2

        #---------------------

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(100, kernel_initializer=tf.keras.initializers.Zeros(), activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(70, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(70, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.model.add(tf.keras.layers.Dense(7))

    def call(self, inputs):
        return self.model(inputs) 

    def loss_labels(self, predicted, labels):
        return tf.reduce_mean(tf.keras.losses.mean_squared_error(predicted, labels))

    def loss(self, inputs, tension):

        pos = inputs[:, :6]
        vel = inputs[:, 6:12]
        end = inputs[:, 12:15]
        end_v = inputs[:, 15:18]
        end_a = inputs[:, 18:21]

        vel_l, acc_l = constrainCalculator(pos, vel, tension[:, :6], end, end_v, end_a, tension[:, 6])

        return tf.reduce_mean(acc_l**2)

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

def pretrain(model, inputs, labels):
    idx = tf.random.shuffle(tf.range(inputs.shape[0]))
    sinputs = tf.gather(inputs, idx)
    slabels = tf.gather(labels, idx)
    loss_sum = 0
    for i in range(0, inputs.shape[0], model.batch_sz):    
        with tf.GradientTape() as tape:
            tension = model(sinputs[i: i+model.batch_sz])
            loss = model.loss_labels(tension, slabels[i: i+model.batch_sz])   
            loss_sum+=loss     
        gradients = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss_sum /(inputs.shape[0] // model.batch_sz)

def pretest(model, inputs, labels):
    sinputs = inputs
    slabels = labels
    loss_sum = 0
    for i in range(0, inputs.shape[0], model.batch_sz):    
       
        tension = model(sinputs[i: i+model.batch_sz])
        loss = model.loss_labels(tension, slabels[i: i+model.batch_sz])   
        loss_sum+=loss     

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
    num_epoch = 10000

    # train_data = np.load('train.npz')
    # test_data = np.load('test.npz')
    # train_inputs, train_labels, train_A, train_w , train_t = train_data['inputs'], train_data['labels'], train_data['A'], train_data['w'], train_data['t']
    # test_inputs, test_labels, test_A, test_w , test_t = test_data['inputs'], test_data['labels'], test_data['A'], test_data['w'], test_data['t']

    # for i in range(num_pre_epoch):
    #     train_loss = pretrain(model, train_inputs, train_labels)
    #     test_loss = pretrain(model, test_inputs, test_labels)

    #     print('Epoch: {:d} train: {:f} test: {:f}'.format(i+1, train_loss, test_loss))
    #     plt.plot(i, train_loss, 'b.', label = 'train loss with labels')
    #     plt.plot(i, test_loss, 'g.', label = 'test loss with labels')

    #     train_loss2 = model.loss(train_inputs, train_labels)
    #     plt.plot(i, train_loss2, 'r.', label = 'train loss w/o labels')
    
    #     plt.pause(0.01)

    #     if i == 0 :
    #         plt.legend()
    #         plt.xlabel('epoch')
    #         plt.ylabel('loss')
    #         plt.yscale('log')

    train_inputs = generate_data(num_train)
    test_inputs = generate_data(num_test)
    
    for i in range(num_pre_epoch, num_pre_epoch + num_epoch):
        # train_loss = train(model, train_inputs)
        train_loss = 0.0
        test_loss = test(model, test_inputs)
        print('Epoch: {:d} train: {:f} test: {:f}'.format(i+1, train_loss, test_loss))
        plt.plot(i, train_loss, 'b.', label = 'train loss')
        plt.plot(i, test_loss, 'g.', label = 'test loss')
    
        plt.pause(0.01)

        if i == 0 :
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')

        if i%10 == 0:
            train_inputs = generate_data(num_train)
            
    

    plt.show()


if __name__ == '__main__':
    main()


    
        
        



