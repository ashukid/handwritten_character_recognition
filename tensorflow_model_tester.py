import tensorflow as tf
import numpy as np
import os
import cv2

import class_label

# getting the map between the labes and corresponding class
# like A mapped with 0, B with 1 etc -> {A:0,B:1,C:2}
label_class_map=class_label.get_class_label()
number_of_classes=len(label_class_map)


def one_hot_convert(y):
    onehot_y = np.zeros(number_of_classes)
    onehot_y[y]=1  
    return onehot_y

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
    W_fc1 = weight_variable([32 * 32 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # Final Output Layer
    W_fc2 = weight_variable([1024, number_of_classes])
    b_fc2 = bias_variable([number_of_classes])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return y_conv


def cnn_computation(x,y_,x_test,y_test):
    y_conv=cnn_graph(x)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predict = tf.argmax(y_conv,1)
    saver=tf.train.Saver()

    with tf.Session() as sess:
    	sess.run(tf.global_variables_initializer())
    	saver.restore(sess=sess,save_path='recognition-model')

    	# use below statement to get accuracy value for test images
    	train_accuracy = accuracy.eval(feed_dict={x:x_test, y_:y_test})
    	print(train_accuracy)

    	# use below statement to get actual predicted value
    	prediction = predict.eval(feed_dict={x:x_test})
    	print(prediction)



def main():

	x_test=np.load('x_test.npy')
	y_test=np.load('y_test.npy')

	print(x_test.shape,y_test.shape)

	new_label=[]
	for i in range(len(y_test)):
		new_label.append(one_hot_convert(y_test[i]))
	y_test=new_label


	x = tf.placeholder(tf.float32,[None,128,128,1],name="x")
	y_= tf.placeholder(tf.float32,[None,number_of_classes],name="y_")

	cnn_computation(x,y_,x_test,y_test)




# main function
if __name__ == '__main__':
    main()


