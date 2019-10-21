import tensorflow as tf

## operation
print(tf.add(1,2)) #덧셈 add
print(tf.add([1,2],[3,4])) #다차원 덧셈
print(tf.square(5)) # 제곱근 square
print(tf.reduce_sum([1,2,3])) #합 reduce_sum -> 텐서의 차원을 탐색하며, 개체들의 총합을 계산

# method overloading 지원됨.
print(tf.square(2) + tf.square(3),end='\n\n') 

x = tf.matmul([[2]],[[2,3]]) #행렬간 곱셈
print(x)
print(x.shape)
print(x.dtype)
print(x.ndim)

print(tf.reshape(x, shape = [2,1]))