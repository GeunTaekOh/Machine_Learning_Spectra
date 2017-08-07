import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100


X = tf.placeholder(tf.float32, [None, 784])		# MNIST 이므로 784개의 열.
X_img = tf.reshape(X,[-1, 28, 28, 1])	# 개수는 모르겠고 (-1), 28 * 28 의 크기의 색상은 1개.  # img : 28*28*1
Y = tf.placeholder(tf.float32, [None, 10])

		## 첫번째 Convolution Layer  ( ? 28, 28, 1) 로 들어올 것임. 
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))		# 3*3 의 필터, 색상은 1개. 32개의 필터사용

L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  
#가온데 1, 1 이 stride 1 칸씩 가겠다 padding은 same이고 stride가 1이므로 결과값이 28 * 28이 될 것임
# convolution 을 통과한 뒤 (?, 28, 28, 32) 개의 모양이 나올 것임.
L1 = tf.nn.relu(L1)	#activation 인 relu 통과.
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
#L1의 이미지를 max pool하는 것임. 이번에는 stride가 2이고, padding이 same이므로 14 * 14가 될것임.
#ksize는 2 * 2 의 입력값에 결과값 1개만을 내보내겠다는 것임
# max pooling을 하면 사진이 조금 찌그러 지지만 외형을 그대로 유지해줌 이렇게 데이터를 축소시켜서 더 효율적으로 이미지를 인식함
# 이 부분을 통과하면 (?, 14, 14, 32)의 모양이 나올 것임.

		## 두번째 Convolution Layer ( ?, 14, 14, 32)로 들어올 것임.
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01)) # 3*3의 필터이고, 32개의 색상을 가진 것을 64개의 filter를 사용할 것이다.
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding = 'SAME')
# (? , 14, 14, 64) 의 모양이 될 것임.
L2 = tf.nn.relu(L2)		#activation function
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')	#padding을 하였고, stride가 2 * 2 이므로 사이즈가 절반 줄게된다.
# (?, 7, 7, 64) 의 모양이 될 것임.
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64]) # -1 : 몇개인지는 모르겠지만 이만큼으로 reshape할 것임. 7*7*64의 모양을!
# fully connected에 넣기 위해서는 7*7*64 의 입체를 평면으로 쭉 펴야 하므로 이를 reshape 해줘야함.
# (? , 3136) 의 벡터가 될 것임. (쭉 펼첬으므로)

		# Fully Connected 부분
W3 = tf.get_variable("W3", shape = [7 * 7 * 64, 10], initializer = tf.contrib.layers.xavier_initializer()) 	#theta값 효율적으로 설정하는 xavier initializer 사용
							# shape은 7*7*64개의 행과 10개의 열이 있음. (10개의 숫자로 분류되야하므로.)
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3)+b
# logits 아 hypothesis 라고 생각하면 됨.

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('learning started. it takes some time.')
for epoch in range(training_epochs):
	avg_cost = 0
	total_batch = int(mnist.train.num_examples / batch_size)

	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		feed_dict = {X:batch_xs, Y : batch_ys}
		c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
		avg_cost += c / total_batch

	print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('learning finished!')

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy : ',sess.run(accuracy, feed_dict = {X:mnist.test.images, Y : mnist.test.labels}))
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
