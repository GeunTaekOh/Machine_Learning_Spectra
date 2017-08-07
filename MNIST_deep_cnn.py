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


keep_prob = tf.placeholder(tf.float32)		# drop out 을 위한 keep_prob

X = tf.placeholder(tf.float32, [None, 784])		# MNIST 이므로 784개의 열.
X_img = tf.reshape(X,[-1, 28, 28, 1])	# 개수는 모르겠고 (-1), 28 * 28 의 크기의 색상은 1개.  # img : 28*28*1
Y = tf.placeholder(tf.float32, [None, 10])

		## 첫번째 Convolution Layer  ( ? 28, 28, 1) 로 들어올 것임. 
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))		# 3*3 의 필터, 색상은 1개. 32개의 필터사용 weight 인 theta 가 filter임.

L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  
#가온데 1, 1 이 stride 1 칸씩 가겠다 padding은 same이고 stride가 1이므로 결과값이 28 * 28이 될 것임
# convolution 을 통과한 뒤 (?, 28, 28, 32) 개의 모양이 나올 것임.
L1 = tf.nn.relu(L1)	#activation 인 relu 통과.
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
#L1의 이미지를 max pool하는 것임. 이번에는 stride가 2이고, padding이 same이므로 14 * 14가 될것임.
#ksize는 2 * 2 의 입력값에 결과값 1개만을 내보내겠다는 것임
# max pooling을 하면 사진이 조금 찌그러 지지만 외형을 그대로 유지해줌 이렇게 데이터를 축소시켜서 더 효율적으로 이미지를 인식함
# 이 부분을 통과하면 (?, 14, 14, 32)의 모양이 나올 것임.
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)


		## 두번째 Convolution Layer ( ?, 14, 14, 32)로 들어올 것임.
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01)) # 3*3의 필터이고, 32개의 색상을 가진 것을 64개의 filter를 사용할 것이다.
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding = 'SAME')
# (? , 14, 14, 64) 의 모양이 될 것임.
L2 = tf.nn.relu(L2)		#activation function
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')	#padding을 하였고, stride가 2 * 2 이므로 사이즈가 절반 줄게된다.
# (?, 7, 7, 64) 의 모양이 될 것임.
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)


		## 세번째 Convolution Layer (?, 7, 7, 64) 로 들어올 것임.
W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev = 0.01)) # 3*3의 필터이고, 64개의 색상을 가진 128개의 filter를 사용한다.
L3 = tf.nn.conv2d(L2, W3, strides = [1,1,1,1], padding = 'SAME')
# (?, 7, 7, 128)
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# (?, 4, 4, 128) 이 될 것임. padding 으로 오른쪽 1칸을 0으로 채웠다고 생까하면 됨.
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)
L3 = tf.reshape(L3, [-1, 4 * 4 * 128])  # 이를 벡터처럼 쭉 펴줌. 

		# 네번째 Convolution Layer (?, 4, 4, 128)로 들어올 것임.
W4 = tf.get_variable("W4", shape = [4 * 4 * 128, 625], initializer = tf.contrib.layers.xavier_initializer()) 	#theta값 효율적으로 설정하는 xavier initializer 사용
							# shape은 4*4*128개의 행과 625개의 열이 있음.
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

		# Fully connected 
W5 = tf.get_variable("W5", shape = [625, 10], initializer = tf.contrib.layers.xavier_initializer())
							# 625개의 input 을 10개의 0~9로 분류해야 하므로 10개의 열
b5 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L4,W5)+b5


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('learning started. it takes some time.')
for epoch in range(training_epochs):
	avg_cost = 0
	total_batch = int(mnist.train.num_examples / batch_size)

	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		feed_dict = {X:batch_xs, Y : batch_ys, keep_prob : 0.7}
		c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
		avg_cost += c / total_batch

	print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('learning finished!')

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy : ',sess.run(accuracy, feed_dict = {X:mnist.test.images, Y : mnist.test.labels, keep_prob:1}))
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob : 1}))
