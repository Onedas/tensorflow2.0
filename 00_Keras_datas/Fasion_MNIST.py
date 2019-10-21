import tensorflow as tf

print('tensorflow version :',tf.__version__) # 2.0.0 2019-10-19

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
print('Fashion mnist dataset on memory')

class_names = {
0:  'T-shirt/top',
1:	'Trouser',
2:	'Pullover',
3:	'Dress',
4:	'Coat',
5:	'Sandal',
6:	'Shirt',
7:	'Sneaker',
8:	'Bag',
9:	'Ankle boot'}

train_x, test_x = train_x /255.0, test_x /255.0

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(5000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(5000).batch(32)

if __name__ == "__main__":
	for images, labels in train_ds:
		print(images.shape, labels.shape)
		break
	
	import matplotlib.pyplot as plt
	plt.imshow(images[0])
	plt.xticks([])
	plt.yticks([])
	plt.xlabel(class_names[labels[0].numpy()])
	plt.show()