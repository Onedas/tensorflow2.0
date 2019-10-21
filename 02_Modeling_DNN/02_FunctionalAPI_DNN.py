# -*- coding: utf-8 -*-
'''
tensorflow version : 2.0.0

@author: Onedas
'''

import tensorflow as tf

#%% define model with functional API
inputs = tf.keras.Input(shape=(28,28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs = inputs, outputs=outputs)
#model.summary()

#%% compile model (optimizer, loss, metrics)
model.compile(optimizer = tf.keras.optimizers.RMSprop(0.0001),
				loss = tf.keras.losses.sparse_categorical_crossentropy,
				metrics = [tf.keras.metrics.sparse_categorical_accuracy])

#%% model train
# need data...
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# train
model.fit(x_train, y_train, epochs = 1)

#%% model evaluate
model.evaluate(x_test, y_test, verbose=2) #verbose=2 : don't print progress bar when you evaluate

#%% model predict
import matplotlib.pyplot as plt
image = x_test[0][tf.newaxis, ...] # shape (1,28,28)
predict = tf.argmax(model.predict(image)[0]).numpy()
## viualize
plt.imshow(tf.squeeze(image))
plt.xticks([])
plt.yticks([])
plt.xlabel('predict : {}, answer : {}'.format(predict, y_test[0]))
plt.show()