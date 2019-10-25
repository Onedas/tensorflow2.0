# -*- coding: utf-8 -*-
'''
tensorflow version : 2.0.0

@author: Onedas
'''

import tensorflow as tf
print('tensorflow version :',tf.__version__) # 2.0.0

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

if __name__ == "__main__":

	for images, labels in train_ds:
		print(images.shape, labels.shape)
		break

	import matplotlib.pyplot as plt
	plt.imshow(tf.squeeze(images[0]))
	plt.xticks([])
	plt.yticks([])
	plt.xlabel(labels[0].numpy())
	plt.show()