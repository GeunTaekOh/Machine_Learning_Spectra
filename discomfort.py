import tensorflow as tf
import numpy as np
# x1 는 기온, x2는 상대습도
def cal_discomfort(x1,x2):
	y_data=[]
	for i, item in enumerate(x1):
		y_data.append(9/5*x1[i] - 0.55 * (1-x2[i]*1/100) * (9/5*x1[i]-26)+32)
	return y_data

xy=np.loadtxt('weather.csv',delimiter=',',dtype=np.float32)
x1_data=xy[:,0:-1]
x2_data=xy[:,[-1]]
y_data=cal_discomfort(x1_data,x2_data)

x1_test=[23.6,21,19.9,20.4,20.5]
x2_test=[59.1,82.5,84.6,64.4,66.9]
y_test=cal_discomfort(x1_test,x2_test)

x1 = tf.placeholder('float')
x2 = tf.placeholder('float')
y = tf.placeholder('float')

w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = x1*w1 + x2*w2+b

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0002)
train = optimizer.minimize(cost)



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(5001):
		if step % 10 ==0:
			cost_val, w_val1,w_val2, _ = sess.run([cost,w1,w2,train],feed_dict={x1:x1_data,x2:x2_data, y:y_data})
			print(step,cost_val, w_val1, w_val2)
	
	x_want_1 = float(input("기온이 얼마인가?"))
	x_want_2 = float(input('상대습도가 얼마인가?'))
	print('불쾌지수는 ',sess.run((x_want_1 * w_val1) + (x_want_2 * w_val2) + b),' 이다.\n')
	#print('test set불쾌지수는 ',sess.run((x1_test * w_val1) + (x2_test * w_val2) + b),' 이다.\n')
	
	prediction = (x1_test * w_val1) + (x2_test * w_val2) + b
	#is_correct = tf.equal(prediction, y_test)
	#accuracy = tf.reduce_mean(abs(prediction - y_test))

	print(sess.run(tf.subtract(prediction,y_test)))
	correct_prediction = tf.subtract(tf.cast(1, 'float'), tf.reduce_mean(tf.subtract(prediction, y_test)))
	accuracy = tf.cast(correct_prediction, "float")

	print('test prediction : ',sess.run(prediction,feed_dict={x1:x1_test, x2:x2_test}))
	print('Accuracy : ',sess.run(accuracy, feed_dict={x1:x1_test, x2:x2_test, y:y_test}))
	print(y_test)



