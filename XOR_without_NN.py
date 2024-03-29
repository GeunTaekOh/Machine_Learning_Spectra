import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype = np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype = np.float32)

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
W=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis>0.5 , dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(10001):
		sess.run(train,feed_dict={X:x_data,Y:y_data})
		if step % 100 ==0:
			print(step,sess.run(cost, feed_dict={X:x_data, Y:y_data}),sess.run(W))

	h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
	print('\nhypothesis : ',h, '\nCorrect : ',c, '\nAccuracy : ',a)

	#Neural Net 을 사용하지 않는 한 일반 학습으로는 XOR 을 구현하기 어려움! 그래프상에서 가설의 선을 긋기 어려우므로!