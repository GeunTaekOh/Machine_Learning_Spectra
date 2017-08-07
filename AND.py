import tensorflow as tf
import numpy as np

learning_rate = 0.01

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)
y_data = np.array([[0],[0],[0],[1]], dtype=np.float32)

X = tf.placeholder(tf.float32, [4,2])
Y = tf.placeholder(tf.float32, [4,1])

# W1 = tf.Variable(tf.random_normal([2,2]), name = 'weight1')
# b1 = tf.Variable(tf.random_normal([2]), name = 'bias1')
# L1 = tf.nn.sigmoid(tf.matmul(X,W1)+b1)

# W2 = tf.Variable(tf.random_normal([2,2]), name = 'weight2')
# b2 = tf.Variable(tf.random_normal([2]), name = 'bias2')
# L2 = tf.nn.sigmoid(tf.matmul(L1,W2)+b2)

# W3 = tf.Variable(tf.random_normal([2,1]), name = 'weight3')
# b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')

W1 = tf.get_variable("W1",shape=[2,4],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([4]), name = 'bias1')
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

W2 = tf.get_variable("W2",shape=[4,4],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([4]), name = 'bias2')
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)


W3 = tf.get_variable("W3",shape=[4,4],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([4]), name = 'bias2')
L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)

W4 = tf.get_variable("W4",shape=[4,1],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([1]), name = 'bias3')



hypothesis = tf.sigmoid(tf.matmul(L3,W4)+b4)
#hypothesis = tf.matmul(L2,W3)+b3

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
#train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis>0.5 , dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(2001):
		sess.run(train,feed_dict={X:x_data,Y:y_data})
		if step % 10 ==0:
			print(step,sess.run(cost, feed_dict={X:x_data, Y:y_data}),sess.run(W4))

	h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
	print('\nhypothesis : ',h, '\nCorrect : ',c, '\nAccuracy : ',a)
	print('theta : ',sess.run(W4))
	print('b : ',sess.run(b4))
