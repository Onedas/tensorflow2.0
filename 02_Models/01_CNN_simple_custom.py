# -*- coding: utf-8 -*-
'''
tensorflow version : 2.0.0

@author: Onedas
'''


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model

import time
#define my cnn model
class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = Conv2D(32, 3, activation ='relu')
		self.maxpool1 = MaxPool2D(pool_size=(2,2))
		self.conv2 = Conv2D(64, 3, activation = 'relu')
		self.maxpool2 = MaxPool2D(pool_size=(2,2))
		self.flatten = Flatten()
		self.d1 = Dense(128, activation='relu')
		self.d2 = Dense(10, activation = 'softmax')

	def call(self, x):
		x= self.conv1(x)
		x= self.maxpool1(x)
		x= self.conv2(x)
		x= self.maxpool2(x)
		x= self.flatten(x)
		x= self.d1(x)
		x= self.d2(x)
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
        
        print ('Time for epoch {} is {:.3%} sec'.format(epoch + 1, time.time()-start))

def test_(test_dataset):
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
      logits = model(x)
      prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
      test_accuracy(prediction, y)
    
    print("테스트 세트 정확도: {:.3%}".format(test_accuracy.result()))



#%%
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
    
    #%% Test
    test_(test_ds)
    
    #%% Predict
    for i in range(1000):
        img, label = x_test[i], y_test[i]
        predict = tf.argmax(tf.squeeze(model(img[tf.newaxis,...])))
        import matplotlib.pyplot as plt
        if predict.numpy() != label:
            plt.imshow(tf.squeeze(img))
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('predict : {}, answer : {}'.format(predict.numpy(),label))
            plt.show()