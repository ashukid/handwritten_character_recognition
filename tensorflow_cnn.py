import tensorflow as tf
import numpy as np
import os

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# this function is used for making the cnn graph.
# tensoflow graphs are static, so we have to first define our graph, with random weights
# and biases. then in cnn_compute() function we'll train and adjust the weights
def cnn_graph(x):
    
    # Resizing the input to 128*128*1
    x_image = tf.reshape(x, [-1, 128, 128, 1])

    # First Convolution Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolution Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connnected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # Final Output Layer
    W_fc2 = weight_variable([1024, number_of_classes])
    b_fc2 = bias_variable([number_of_classes])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return y_conv



# this function will be used for training the graph we made in cnn_graph function
def cnn_computation(y_conv):
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver() 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            sess.run(train_step,feed_dict={x:train, y_:labels})
            train_accuracy = accuracy.eval(feed_dict={x:train, y_:labels})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        y_pred = tf.nn.softmax(y_conv,name="y_pred")
        saver.save(sess,'cats-dogs-model')
        return y_pred


def main():
    train=np.load('train.npy')
    labels=np.load('label.npy')





# main function
if __name__ == '__main__':
    main()




        
