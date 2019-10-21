import tensorflow as tf

## constant : 상수
helloworld =tf.constant("hello, world")
print("Tensor 	: ",helloworld)
print("Value 	: ",helloworld.numpy()) #numpy() 로 numpy형태로 변환가능하다

