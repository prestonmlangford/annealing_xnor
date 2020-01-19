from __future__ import print_function

import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda, InputLayer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Layer
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def varsoftsign(x,g): return x/(g+K.abs(x))

class AnnealingSign(Layer):


    def __init__(self, output_dim, gamma, **kwargs):
        self.output_dim = output_dim
        self.gamma = gamma
        super(AnnealingSign, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        self.b = K.ones(shape=input_shape)
        super(AnnealingSign, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        self.add_loss(K.sum(1/K.square(self.gamma+K.abs(self.kernel))+self.gamma*K.square(self.kernel))/(self.kernel.shape[0]*self.kernel.shape[1]))
        #xb = K.concatenate([x,self.b],axis=-1)
        w = varsoftsign(self.kernel,self.gamma)
        z = K.dot(x, w)#K.dot(xb, w)
        y = varsoftsign(z,self.gamma)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

#get_custom_objects().update({'variable_softsign': Activation(variable_softsign)})


num_classes = 10
epochs = 20

# the data, split between train and test sets
# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


gamma = K.variable(
    value=1,
    dtype='float32',
    name='gamma',
)
gamma._trainable = False

model = Sequential()
model.add(InputLayer((784,), batch_size=50))
model.add(AnnealingSign(512,gamma,))
#model.add(Dense(512, activation='variable_softsign', input_shape=(784,)))
#model.add(Dropout(0.2))
model.add(AnnealingSign(512,gamma))
model.add(AnnealingSign(512,gamma))
#model.add(Dense(512, activation='variable_softsign'))
#model.add(Dropout(0.2))
model.add(AnnealingSign(num_classes,gamma))
#model.add(Lambda(lambda x: (x+1)/2))
#model.add(Dense(num_classes, activation='softmax'))

model.summary()


model.compile(
    loss='categorical_hinge',
    optimizer=Adam(),
    metrics=['accuracy']
)


plt.ion()
plt.show()

def hist_weights(model):
    w = np.array([])
    for arr in model.get_weights():        
        # We can set the number of bins with the `bins` kwarg
        w = np.concatenate((w,arr.flatten()))
    w = varsoftsign(w.astype('float32'),gamma)
    plt.cla()
    plt.hist(w, bins=1000)
    plt.draw()
    plt.pause(0.1)

anneal = 5
for i in range(25):
    if i > anneal:
        K.set_value(gamma,math.exp(-0.4*(i-anneal)))
    print("gamma = " + str(gamma))
    history = model.fit(x_train, y_train,
                    epochs=1,
                    verbose=1,
                    validation_data=(x_test, y_test))
    hist_weights(model)
    
    

n = 20
for i in range(n+1):
    if i == n:
        K.set_value(gamma,0)
    else:
        K.set_value(gamma,math.pow(0.1,i))
    print(gamma)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])