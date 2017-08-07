import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

##  classification 으로 mnist 분리 하고 accuracy 최대한 높이는 방법
##  Xavier 로 적절한 theta 값을 찾고 Deep NN 을 사용함, Wide NN 을 사용함.
##  -> 하지만 오히려 Wide NN 을 하고 더 깊게하면 accuracy 가 조금 더 떨어짐. 이는! 각 데이터마다 다양한 요인이 있을 것인데 이 데이터에서는 아마 overfitting 일 것임.
## -> 학습을 너무 깊게 하면 training data 에 대해 너무 잘 공부해서 overfitting 이 되어서 새로운 값을 예측하는 대에는 조금 안좋을 수도 있음.
## -> 하지만 이 overfitting 문제는 dropout 으로 해결할 수 있음!!

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)		# keep_prob는 이는 얼마나 drop out 을 할지에 대한 것임 학습을 할때는 보통 0.5~0.7을 하고 실전을 할 때는 1.0 으로 해야함
											# 이 의미는 학습을 할때 데이터를 0.5~0.7만큼사용하지만 실제 테스팅할때는 모든 노드들을 다 사용해야하므로 1.0으로 해주어야함
											# overfitting 문제 때문에 drop out을 함

# weights & bias for nn layers
#W1 = tf.Variable(tf.random_normal([784, 256]))
# theta 값을 랜덤으로 말고 xavier 를 사용해서 적절한 theta 값을 찾음  -> 첫번째 epoch 부터 아예 cost가 낮음!! 나름 이미 적절한 theta 값으로 initialize가 됨.
W1 = tf.get_variable("W1",shape=[784,512],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1,keep_prob=keep_prob)

W2 = tf.get_variable("W2",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2,keep_prob=keep_prob)

W3 = tf.get_variable("W3",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
L3 = tf.nn.dropout(L3,keep_prob=keep_prob)

W4 = tf.get_variable("W4",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
L4 = tf.nn.dropout(L4,keep_prob=keep_prob)

W5 = tf.get_variable("W5",shape=[512,10],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4,W5)+b5



# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer 도 GD 말고 다양한 optimizer 방법이 있는데 어떤 방법이 더 잘되는지를 확인할 수 있음
# 통상적으로 AdamOptimizer 가 보통 좋음!!

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob :0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob : 1}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob:1}))

plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
