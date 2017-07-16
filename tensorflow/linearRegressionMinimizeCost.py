import tensorflow as tf
import matplotlib.pyplot as plt

"""X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

sess.run(tf.initialize_all_variables())

W_val = []
cost_val = []

for i in range(-30, 50) :
  feed_W = i * 0.1
  curr_cost, curr_W = sess.run([cost, W], feed_dict = {W: feed_W})
  W_val.append(curr_W)
  cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()
"""

"""
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_sum(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()

sess.run(tf.initialize_all_variables())

for step in range(40) :
  sess.run(update, feed_dict = {X : x_data, Y : y_data})
  print(step, sess.run(cost, feed_dict = {X : x_data, Y : y_data}), sess.run(W))
"""


X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.)

hypothesis = W * X

gradient = tf.reduce_mean((W * X - Y) * X) * 2

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

gvs = optimizer.compute_gradients(cost)

apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(100) :
  print(step, sess.run([gradient, W, gvs]))
  sess.run(apply_gradients)
