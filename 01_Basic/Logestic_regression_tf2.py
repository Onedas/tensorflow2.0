import tensorflow as tf
'''
simple Logestic regression 

z= wx+b
y_hat = 1/(1+e**-z)

loss = y*log(y_hat) + (1-y) * log(1-y_hat) : binary cross entropy

'''
### Parameters
class Logestic_regression:
	def __init__(self):
		self.W = tf.Variable(10.0)
		self.b = tf.Variable(-3.0)

	def __call__(self, x):
		return 1/(1+tf.math.exp(-self.W * x - self.b))

model = Logestic_regression()

def loss(y_hat, y):
	# return -y* tf.math.log(y_hat) -(1-y)*(1-tf.math.log(1-y_hat))
	return tf.reduce_mean(tf.reduce_sum(tf.losses.binary_crossentropy(y, y_hat)))
	# return tf.reduce_mean(tf.square(y_hat - y))

### 데이터셋 만들기
sample_W = 5
sample_b = 4
Num_examples = 100

Xs = tf.random.normal(shape=[Num_examples])
noise = tf.random.normal(shape=[Num_examples])
Ys = tf.cast(1/(1+tf.math.exp(Xs * sample_W + sample_b + noise)) > 0.5, dtype = float) #cast : type 변환

def train(model, inputs, outputs, learning_rate=0.1):
	with tf.GradientTape() as t:
		current_loss = loss(model(inputs), outputs)

	dW, db = t.gradient(current_loss, [model.W, model.b])
	model.W.assign_sub(learning_rate * dW) #assign_sub : 값을 감소. -=과 같음
	model.b.assign_sub(learning_rate * db) 

model = Logestic_regression()
  # 도식화를 위해 W값과 b값의 변화를 저장.
Ws, bs = [], []
epochs = range(10000)
for epoch in epochs:
	Ws.append(model.W.numpy())
	bs.append(model.b.numpy())
	current_loss = loss(model(Xs), Ys)

	train(model, Xs, Ys, learning_rate=0.1)
	if epoch % 100 == 0:
		print('{:5} epoch ==> loss : {:7.5f}\t y={:.3f}x+{:.3f}'.format(epoch,current_loss,Ws[-1],bs[-1]))

import matplotlib.pyplot as plt
fig= plt.figure()
plt.scatter(Xs, Ys, c='b')
plt.scatter(Xs, model(Xs), c='r')
plt.show()
