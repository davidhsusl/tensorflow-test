import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# print(mnist.train.images.shape, mnist.train.labels.shape)
# print(mnist.test.images.shape, mnist.test.labels.shape)
# print(mnist.validation.images.shape, mnist.validation.labels.shape)

W = tf.Variable([.3], dtype=tf.float32, name="W")
b = tf.Variable([-.3], dtype=tf.float32, name="b")
x = tf.placeholder(dtype=tf.float32, name="x")

with tf.name_scope('linear_model'):
    linear_model = W * x + b

y = tf.placeholder(tf.float32, name="y")

square_deltas = tf.square(linear_model - y)
with tf.name_scope('loss'):
    loss = tf.reduce_sum(square_deltas)
    tf.summary.scalar("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
    # 匯出流程圖
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./tfboard', sess.graph)

    sess.run(tf.global_variables_initializer())

    # print(sess.run(loss, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    for i in range(1000):
        summary, _ = sess.run([merged, train_step], feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
        train_writer.add_summary(summary, i)

        if i % 10 == 0:
            print('count: %s' % i)
            # print(sess.run(W))
            # print(sess.run(b))
            # print(sess.run(linear_model, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
            # print(sess.run(square_deltas, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
            print('loss: %s' % sess.run(loss, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    print(sess.run([W, b]))
    train_writer.close()