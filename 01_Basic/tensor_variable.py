import tensorflow as tf

## Variable : 변수
variable = tf.Variable([1,2,3])
print("Tensor 	: {}".format(variable))
print("value 	: {}".format(variable.numpy()))