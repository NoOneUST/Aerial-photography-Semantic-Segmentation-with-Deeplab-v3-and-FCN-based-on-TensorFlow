from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import time

batch_size=64
max_epoch=20
pooling_size=2

def lenet(x):
  with tf.name_scope('reshape'):
    image2d = tf.reshape(x, [-1, 28, 28, 1])

  with tf.name_scope('conv1'):
    W_conv1 = weight([5, 5, 1, 32])
    b_conv1 = bias([32])
    o_conv1 = tf.nn.relu(conv2d(image2d, W_conv1) + b_conv1)

  with tf.name_scope('pool1'):
    o_pool1 = max_pool(o_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight([5, 5, 32, 64])
    b_conv2 = bias([64])
    o_conv2 = tf.nn.relu(conv2d(o_pool1, W_conv2) + b_conv2)

  with tf.name_scope('pool2'):
    o_pool2 = max_pool(o_conv2)

  with tf.name_scope('fc1'):
    feature_count = 7 * 7 * 64
    W_fc1 = weight([feature_count, 1024])
    b_fc1 = bias([1024])

    o_pool2_1D = tf.reshape(o_pool2, [-1, feature_count])
    o_fc1 = tf.nn.relu(tf.matmul(o_pool2_1D, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    o_fc1_drop = tf.nn.dropout(o_fc1, keep_prob)

  with tf.name_scope('fc2'):
    W_fc2 = weight([1024, 10])
    b_fc2 = bias([10])

    y_predict = tf.matmul(o_fc1_drop, W_fc2) + b_fc2
  return y_predict, keep_prob


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, pooling_size, pooling_size, 1],
                        strides=[1, pooling_size, pooling_size, 1], padding='SAME')


def weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main():
  start_time = time.time()
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784])

  y_ = tf.placeholder(tf.float32, [None, 10])

  y_predict, keep_prob = lenet(x)

  with tf.name_scope('loss'):
    cross_entropy_matrix = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_predict)
  cross_entropy = tf.reduce_mean(cross_entropy_matrix)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correction_verify_bool = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_, 1))
    correction_verify_bin = tf.cast(correction_verify_bool, tf.float32)

  accuracy = tf.reduce_mean(correction_verify_bin)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(math.floor(mnist.train.num_examples*max_epoch/batch_size)):
      data_this_batch = mnist.train.next_batch(batch_size)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: data_this_batch[0], y_: data_this_batch[1], keep_prob: 1.0})
        print(('step %d, training accuracy %g' % (i, train_accuracy)))
      train_step.run(feed_dict={x: data_this_batch[0], y_: data_this_batch[1], keep_prob: 0.5})

    print(('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
    print(time.time() - start_time)

if __name__ == '__main__':
    main()