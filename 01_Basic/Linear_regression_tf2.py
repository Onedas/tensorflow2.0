import tensorflow as tf
'''
f(x) = x* W + b

'''
### Linear regression 모델 정의
class Linear_regression(object):
	def __init__(self):

		self.W = tf.Variable(5.0)
		self.b = tf.Variable(3.0)

	def __call__(self, x):
		return self.W * x + self.b

model = Linear_regression()

### loss 함수 정의
def loss(y_hat, y):

	return tf.reduce_mean(tf.square(y_hat - y))

### 데이터셋 만들기 m
True_W = 3.0
True_b = 2.0
Num_examples = 1000

Xs = tf.random.normal(shape=[Num_examples])
noise = tf.random.normal(shape=[Num_examples])
Ys = Xs * True_W + True_b + noise

### 시각화
import matplotlib.pyplot as plt
fig= plt.figure()
plt.scatter(Xs, Ys, c='b')
plt.scatter(Xs, model(Xs), c='r')
plt.show()

print('현재 손실 : ',loss(model(Xs),Ys).numpy())

## 훈련 루프
'''
경사 하강법에는 여러가지 방법이 있으며, 
tf.train.Optimizer 에 구현되어있음. 
'''
def train(model, inputs, outputs, learning_rate=0.1):
	with tf.GradientTape() as t:
		current_loss = loss(model(inputs), outputs)

	dW, db = t.gradient(current_loss, [model.W, model.b])
	model.W.assign_sub(learning_rate * dW) #assign_sub : 값을 감소. -=과 같음
	model.b.assign_sub(learning_rate * db) 

model = Linear_regression()
  # 도식화를 위해 W값과 b값의 변화를 저장.
Ws, bs = [], []
epochs = range(100)
for epoch in epochs:
	Ws.append(model.W.numpy())
	bs.append(model.b.numpy())
	current_loss = loss(model(Xs), Ys)

	train(model, Xs, Ys, learning_rate=0.1)
	print('{:5} epoch ==> loss : {:7.5f}\t y={:.3f}x+{:.3f}'.format(epoch,current_loss,Ws[-1],bs[-1]))

### 도식화
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([True_W] * len(epochs), 'r--',
         [True_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()

####################
