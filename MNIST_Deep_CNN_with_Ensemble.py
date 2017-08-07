import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 20
batch_size = 100

class Model:
	# 초기화부분
	def __init__(self, sess, name):
		self.sess = sess
		self.name = name
		self._build_net()


	# 네트워크 구성하는 함수
	def _build_net(self):
		with tf.variable_scope(self.name):
 			self.training = tf.placeholder(tf.bool)

 			self.X = tf.placeholder(tf.float32, [None, 784])

 			X_img = tf.reshape(self.X, [-1, 28, 28, 1])
 			self.Y = tf.placeholder(tf.float32, [None,10])

 			## Convolutional Layer 1
 			conv1 = tf.layers.conv2d(inputs = X_img, filters = 32, kernel_size = [3,3], padding = "SAME", activation = tf.nn.relu)
 			pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], padding = "SAME", strides = 2)
 			dropout1 = tf.layers.dropout(inputs = pool1, rate = 0.7, training = self.training)

 			## Convolutional Layer2, Polling Layer2
 			conv2 = tf.layers.conv2d(inputs=dropout1, filters = 64, kernel_size = [3,3], padding="SAME", activation = tf.nn.relu)
 			pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], padding = "SAME", strides = 2)
 			dropout2 = tf.layers.dropout(inputs = pool2, rate = 0.7, training = self.training)

 			## Convolutional Layer3, Polling Layer3
 			conv3 = tf.layers.conv2d(inputs=dropout2, filters = 128, kernel_size=[3,3], padding = "SAME", activation = tf.nn.relu)
 			pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size=[2,2], padding="SAME", strides=2)
 			dropout3 = tf.layers.dropout(inputs = pool3, rate = 0.7, training = self.training)

 			## Dense Layer with Relu
 			flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])
 			dense4 = tf.layers.dense(inputs = flat, units = 625, activation = tf.nn.relu)  #dense 는 fully connected network 를 만드는 api 
 			dropout4 = tf.layers.dropout(inputs = dense4, rate = 0.5, training=self.training)

 			# Logits Layer
 			self.logits = tf.layers.dense(inputs=dropout4, units=10)

 		#define cost/loss & optimizer
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

		correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def prediction(self,x_test, training = False):
		return self.sess.run(self.logits, feed_dict = {self.X : x_test, self.training : training})

	def get_accuracy(self, x_test, y_test, training = False):
		return self.sess.run(self.accuracy, feed_dict={self.X : x_test, self.Y : y_test, self.training : training})

	def train(self, x_data, y_data, training =True):
 		return self.sess.run([self.cost, self.optimizer], feed_dict={self.X:x_data, self.Y : y_data, self.training : training})


sess = tf.Session()

models = []
num_models = 2

for m in range(num_models):
	models.append(Model(sess,"model"+str(m)))

sess.run(tf.global_variables_initializer())

print('Learning started!')


for epoch in range(training_epochs):
	avg_cost_list = np.zeros(len(models))
	total_batch = int(mnist.train.num_examples / batch_size)
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)

		for m_idx, m in enumerate(models):
			c, _ = m.train(batch_xs, batch_ys)
			avg_cost_list[m_idx] += c / total_batch
	print('epoch : ','%04d' % (epoch +1), 'cost = ',avg_cost_list)

print('learning finished!!')

test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)
for m_idx, m in enumerate(models):
	print(m_idx, 'Accuracy : ', m.get_accuracy(mnist.test.images, mnist.test.labels))
	p = m.predict(mnist.test.images)
	predictions += p  # 모든 model 에서의 각 0~9까지의 어떤값일지에 대한 확률값을 다 합친 것.

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))
