import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype = np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype = np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2,2]),name='weight1')
b1 = tf.Variable(tf.random_normal([2]),name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.random_normal([2,1]),name='weigth2')
b2 = tf.Variable(tf.random_normal([1]),name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)

#이렇게 쌓는것을 깊게 (w3,b3, w4,b4 로) 하면 deep NN 이고
#이를 [2,10], [10,1] 로 하면 wide NN 가 된다. 
#둘다 학습이 더 잘되지만 크게 의미는 없음 . 어차피 sigmoid 로 0,1 로 판단하는대
#판단하기 전 hypothesis 값이 더 1과 가깝고 0과 가깝게 되는 것 뿐임


cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5 , dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y),dtype = tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(10001):
		sess.run(train,feed_dict={X:x_data, Y:y_data})
		if step % 100 == 0:
			print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}),sess.run([W1,W2]))

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
	print('\nhypothesis : ',h, '\ncorrect : ',c,'\nAccuracy : ',1)
