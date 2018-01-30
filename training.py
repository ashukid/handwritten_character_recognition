import tensorflow as tf
import numpy as np
import os
import cv2

# this function is used to convert the images to 128*128 tensors
def image_to_tensor():
    image_path="train_dataset"
    image_path = os.getcwd() + "/" + image_path
    file_reader=os.listdir(image_path)

    train=[]
    label=[]
    class_label_map={}
    class_number=-1
    for folders in file_reader:
        if(os.path.isdir(os.path.join(image_path,folders))):
            image_reader=os.listdir(os.path.join(image_path,folders))
            path=os.path.join(image_path,folders)
            class_number+=1
            class_label_map[class_number]=folders
            for image in image_reader:
                if(image.lower().endswith(('.png', '.jpg', '.jpeg'))):
                    # parameter 0 for grayscale, 1 for RGB
                    train.append(cv2.imread(os.path.join(path,image),0))
                    label.append(class_number)


    # training data and labels converted from list to numpy arrays and saved on the disk
    train=np.array(train)
    train=train.reshape(-1,128,128,1) 
    label=np.array(label)

    return (train,label,class_label_map)



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


# this function is used for making the cnn graph.
# tensoflow graphs are static, so we have to first define our graph, with random weights
# and biases. then in cnn_compute() function we'll train and adjust the weights
def cnn_graph(x):
    
    # Resizing the input to 128*128*1
    x_image = tf.reshape(x, [-1, 128, 128, 1])

    # First Convolution Layer
    W_conv1 = weight_variable([5, 5, 1, 8])
    b_conv1 = bias_variable([8])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolution Layer
    W_conv2 = weight_variable([5, 5, 8, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connnected Layer
    W_fc1 = weight_variable([32 * 32 * 16, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # Final Output Layer
    W_fc2 = weight_variable([256, number_of_classes])
    b_fc2 = bias_variable([number_of_classes])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return y_conv



# this function will be used for training the graph we made in cnn_graph function
def cnn_computation(x,y_,x_train,y_train,epoch,batch_size):

    # getting the static graph with random weights and biases
    y_conv=cnn_graph(x)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver() 

    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logdir/default_graph", graph=tf.get_default_graph())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(epoch):

            i=0
            while i < len(x_train):
                start = i
                end = i+batch_size
                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])

                sess.run(optimizer,feed_dict={x:batch_x, y_:batch_y})
                i+=batch_size

            # updating the tensorboard summatry for accuracy and entropy loss
            summary= sess.run(summary_op,feed_dict={x:x_train, y_:y_train})
            writer.add_summary(summary,j)

            # calculating the accuracy after j epochs
            train_accuracy = accuracy.eval(feed_dict={x:x_train, y_:y_train})
            print('After step %d, training accuracy %g' % (j, train_accuracy))


        saver.save(sess,os.path.join(os.getcwd(), 'recognition-model'))


def main(train,label):

    print(train.shape,label.shape)

    # train = train.astype('float32')
    # labels = labels.astype('float32')

    # converting labels into one hot
    # for example : [2,1] -> [[0,0,1],[0,1,0]]
    new_label=[]
    for i in range(len(label)):
        new_label.append(one_hot_convert(label[i]))
    label=new_label

    # hyperparameters
    epoch=2
    batch_size=5000

    # defining the placeholders to be replaced by valued during the session
    # size of x = 128*128*1, 1 for grayscale value
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,[None,128,128,1],name="x")
        y_= tf.placeholder(tf.float32,[None,number_of_classes],name="y_")

    # calling the computation graph, passing all the user defined parameters
    cnn_computation(x,y_,train,label,epoch,batch_size)




# main function
if __name__ == '__main__':
    train,label,class_label_map=image_to_tensor()
    number_of_classes=len(class_label_map)
    main(train,label)


