# -*- coding: utf-8 -*-
'''
tensorflow version : 2.0.0

@author: Onedas
'''

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

import time
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(64, activation = 'relu')
        self.d3 = Dense(10, activation = 'softmax')
	
    def call(self, x):
        x= self.flatten(x)
        x= self.d1(x)
        x= self.d2(x)
        x= self.d3(x)
        return x

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predicts = model(images)
        loss = loss_object(labels, predicts)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip (gradients, model.trainable_variables))
    
def train(dataset, epochs):
    
    for epoch in range(epochs):
        start = time.time()
        
        for images, labels in dataset:
            train_step(images,labels)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


if __name__ == "__main__":
    print('tensorflow version :',tf.__version__)
       
    #%% load mnist_data
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    #%% Model, loss, optimizer  
    model = MyModel()
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    
    #%% train
    train(train_ds, 5)
