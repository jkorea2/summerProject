import tensorflow as tf

""""hello = tf.constant('Hello, TensorFlow!!')

sess = tf.Session()

print(sess.run(hello))""" # Hello World

"""node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) #node3 = node1 + node2


sess = tf.Session()
print("sess.run(node1, node2) = ", sess.run([node1, node2]))
print("sess.run(node3) = ", sess.run(node3))""" # node1 + node2


"""a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adderNode = a + b

print(sess.run(adderNode, feed_dict = {a: 3, b: 4.5}))
print(sess.run(adderNode, feed_dict = {a: [1, 3], b:[2, 6]}))""" # node plus with variable

""" Linear Regression with Train Set
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
B = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + B

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(2001):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(cost), sess.run(W), sess.run(B))

"""



W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(2001):
  cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
    feed_dict = {X : [1, 2, 3, 4, 5],
      Y: [2.1, 3.1, 4.1, 5.1, 6.1]})

  if step % 20 == 0:
    print(step, cost_val, W_val, b_val)
