import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#ex1.
xy = np.loadtxt('OnlineNewsPopularity.csv',delimiter=',',dtype=np.float32,skiprows=1, usecols=range(2,61))

xy_train, xy_tmp = train_test_split(xy, test_size=0.2, random_state=42)
xy_validation, xy_test = train_test_split(xy_tmp, test_size=0.5, random_state=42)

#80% training data
x_data=xy_train[:,0:-1]
y_data=xy_train[:,[-1]]

print('x_data : ',x_data)
print('\n')
print('y_data : ',y_data)

# 10% validation data
x_validation = xy_validation[:,0:-1]
y_validation = xy_validation[:,[-1]]
print('x_validation : ',x_validation)
print('\n')
print('y_validation : ',y_validation)

# 10% test data
x_test = xy_test[:,0:-1]
y_test = xy_test[:,[-1]]
print('x_test : ',x_test)
print('\n')
print('y_test : ',y_test)


X = tf.placeholder('float',[None,58])
y = tf.placeholder('float',[None,1])
alpha = tf.placeholder('float')

#Multiple Variable Linear Regression
w = tf.Variable(tf.random_normal([58,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

X_normalized = (X - tf.reduce_mean(X,axis=0)) / (tf.reduce_max(X,axis=0) - tf.reduce_min(X,axis=0))
hypothesis = tf.matmul(X_normalized,w)+b
#hypothesis = tf.matmul(X,w)+b

cost = (tf.reduce_mean(tf.square(hypothesis-y))) * 0.5
optimizer=tf.train.GradientDescentOptimizer(learning_rate=alpha)
train = optimizer.minimize(cost)

#Polynomial Linear Regression
w_2 = tf.Variable(tf.random_normal([116,1]), name='weight2')
b_2 = tf.Variable(tf.random_normal([1]), name='bias2')

X_square = tf.square(X)
X_2 = tf.concat([X, X_square], axis=1)
X_2_normalized = (X_2 - tf.reduce_mean(X_2,axis=0)) / (tf.reduce_max(X_2, axis=0) - tf.reduce_min(X_2, axis=0))
hypothesis_2 = tf.matmul(X_2_normalized,w_2) + b_2

lamda = 0.01
cost_2 = (tf.reduce_mean(tf.square(hypothesis_2 - y))  + lamda * tf.reduce_mean(tf.square(w_2)))*0.5
optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train_2 = optimizer_2.minimize(cost_2)

#learning_rate_array = [0.01, 0.03, 0.1, 0.3]
learning_rate_array = [0.3]
#learning_rate_array = [1e-12]

sess = tf.Session()
for a in learning_rate_array:
    sess.run(tf.global_variables_initializer())
    cost_history = []
    step_history = []
    for step in range(201):
        cost_val, hy_val, _ = sess.run([cost_2,hypothesis_2,train_2],feed_dict={X:x_data,y:y_data, alpha:a})
        #cost_val, hy_val, _ = sess.run([cost,hypothesis,train],feed_dict={X:x_data,y:y_data, alpha:a})
        cost_history.append(cost_val)
        step_history.append(step)
        if step % 20 == 0:
            #print(step,'cost : ',cost_val, '\nPrediction : \n')#, '\nPrediction : \n',hy_val,'\n')
            print(step,'cost : ',cost_val, '\nPrediction : \n',hy_val,'\n')
            #for i in range(len(hy_val)):
            #    print('',i,' ',hy_val[i],'\n')
            #    print('\n')
    plt.plot(step_history, cost_history, label=( 'learning rate = ' + str(a)))

print('VALIDATION 10%\n')
for step in range(201):
    valid_cost,hy_valid,_ = sess.run([cost_2,hypothesis_2,train_2], feed_dict={X: x_validation, y: y_validation, alpha: 0.3})
    if step%20==0:
        print(step,'cost : ',valid_cost, '\nPrediction : \n',hy_valid,'\n')


print('TEST 10%\n')
for step in range(201):
    test_cost,hy_test,_ = sess.run([cost_2,hypothesis_2,train_2], feed_dict={X: x_test, y: y_test, alpha: 0.3})
    if step % 20==0:
        print(step,'cost : ',test_cost, '\nPrediction : \n',hy_test,'\n')


print('In learning rate : 0.3\n Validation Cost : %lf, Test Cost : %lf ' % (valid_cost, test_cost))

plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.legend()
plt.show()



## ex2.
## Use Only important theta values in Polynimial Linear Regression with Regularization
xy_mo = np.loadtxt('modifiedOnline.csv',delimiter=',',dtype=np.float32,skiprows=1,usecols=range(0,23))

print('xy_mo : ',xy_mo)

xy_train_mo, xy_tmp_mo = train_test_split(xy_mo, test_size=0.2, random_state=42)
xy_validation_mo, xy_test_mo = train_test_split(xy_tmp_mo, test_size=0.5, random_state=42)

# 80% training data
x_test_data_mo=xy_train_mo[:,0:-1]
y_test_data_mo=xy_train_mo[:,[-1]]
print('x_data : ',x_test_data_mo)
print('\n')
print('y_data : ',y_test_data_mo)


# 10% validation data
x_validation_mo = xy_validation_mo[:,0:-1]
y_validation_mo = xy_validation_mo[:,[-1]]
print('x_validation : ',x_validation_mo)
print('\n')
print('y_validation : ',y_validation_mo)

# 10% test data
x_test_mo = xy_test_mo[:,0:-1]
y_test_mo = xy_test_mo[:,[-1]]
print('x_test : ',x_test_mo)
print('\n')
print('y_test : ',y_test_mo)

X_mo = tf.placeholder('float',[None,22])
y_mo = tf.placeholder('float',[None,1])

w_mo = tf.Variable(tf.random_normal([44,1]),name='weight')
b_mo = tf.Variable(tf.random_normal([1]),name='bias')

X_square_mo = tf.square(X_mo)
X_2_mo = tf.concat([X_mo, X_square_mo], axis=1)

X_2_normalized_mo = (X_2_mo - tf.reduce_mean(X_2_mo,axis=0)) / (tf.reduce_max(X_2_mo, axis=0) - tf.reduce_min(X_2_mo, axis=0))
hypothesis_2_mo = tf.matmul(X_2_normalized_mo,w_mo) + b_mo

lamda = 0.01
cost_2_mo = (tf.reduce_mean(tf.square(hypothesis_2_mo - y_mo))  + lamda * tf.reduce_mean(tf.square(w_mo)))*0.5
optimizer_2_mo = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train_2_mo = optimizer_2_mo.minimize(cost_2_mo)

sess.run(tf.global_variables_initializer())
cost_history_mo = []
step_history_mo = []

for step in range(201):
    cost_val_mo, hy_val_mo, _ = sess.run([cost_2_mo,hypothesis_2_mo,train_2_mo],feed_dict={X_mo:x_test_data_mo,y_mo:y_test_data_mo, alpha:0.3})
    cost_history_mo.append(cost_val_mo)
    step_history_mo.append(step)
    if step % 20 == 0:
        print(step,'cost : ',cost_val_mo, '\nPrediction : \n',hy_val_mo,'\n')

plt.plot(step_history_mo, cost_history_mo, label=( 'learning rate = 0.3'))


print('VALIDATION 10%\n')
valid_cost_mo,hy_validation = sess.run([cost_2_mo,hypothesis_2_mo], feed_dict={X_mo: x_validation_mo, y_mo: y_validation_mo, alpha: 0.3})
print(valid_cost_mo)
print(hy_validation)


print('TEST 10%\n')
test_cost_mo,hy_test = sess.run([cost_2_mo,hypothesis_2_mo], feed_dict={X_mo: x_test_mo, y_mo: y_test_mo, alpha: 0.3})
print(test_cost_mo)
print(hy_test)

print('In learning rate : 0.3\n Validation Cost : %lf, Test Cost : %lf ' % (valid_cost_mo, test_cost_mo))
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.legend()
plt.show()