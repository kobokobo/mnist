import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
## from tensorflow.python import debug as tf_debug


# reading MNIST data via tensorflow function
mnist = input_data.read_data_sets("mnist/", one_hot=True)

pixels = 28 * 28 # 28x28 pixel
nums = 10 # 10 class(0-9 digit)
bit_length = 15

# label # pn-code -------------------------------------------->
# 0# -1  -1  -1   1  -1  -1   1   1  -1   1  -1   1   1   1   1
# 1# -1  -1   1  -1  -1   1   1  -1   1  -1   1   1   1   1  -1
# 2# -1   1  -1  -1   1   1  -1   1  -1   1   1   1   1  -1  -1
# 3#  1  -1  -1   1   1  -1   1  -1   1   1   1   1  -1  -1  -1
# 4# -1  -1   1   1  -1   1  -1   1   1   1   1  -1  -1  -1   1
# 5# -1   1   1  -1   1  -1   1   1   1   1  -1  -1  -1   1  -1
# 6#  1   1  -1   1  -1   1   1   1   1  -1  -1  -1   1  -1  -1
# 7#  1  -1   1  -1   1   1   1   1  -1  -1  -1   1  -1  -1   1
# 8# -1   1  -1   1   1   1   1  -1  -1  -1   1  -1  -1   1   1
# 9#  1  -1   1   1   1   1  -1  -1  -1   1  -1  -1   1   1  -1
pncode0 = np.array([-1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1], dtype = 'float')
pncode1 = np.array([-1,  -1,   1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1], dtype = 'float')
pncode2 = np.array([-1,   1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1], dtype = 'float')
pncode3 = np.array([ 1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1], dtype = 'float')
pncode4 = np.array([-1,  -1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1], dtype = 'float')
pncode5 = np.array([-1,   1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1], dtype = 'float')
pncode6 = np.array([ 1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1,  -1], dtype = 'float')
pncode7 = np.array([ 1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1,  -1,   1], dtype = 'float')
pncode8 = np.array([-1,   1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1,  -1,   1,   1], dtype = 'float')
pncode9 = np.array([ 1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1], dtype = 'float')

# declare of  placeholder
x  = tf.placeholder(tf.float32, shape=(None, pixels), name="x") # image data
y_ = tf.placeholder(tf.float32, shape=(None, bit_length), name="y_")  # label (teaching data pn)
pn_ = tf.placeholder(tf.float32, shape=(bit_length, nums), name="pn_")  # all label for pn
y_org = tf.placeholder(tf.float32, shape=(None, nums), name="y_org")  # original label (teaching data)

# initialize for weight and bais [-0.1 ~ 0.1] gauss distribution
def weight_variable(name, shape):
    W_init = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(W_init, name="W_"+name)
    return W
def bias_variable(name, size):
    b_init = tf.constant(0.1, shape=[size])
    b = tf.Variable(b_init, name="b_"+name)
    return b

# convolution function
def conv2d(x, W):
    # strides=[1,1,1,1]
    # HWCN: 1 stride on image hight, 1 stride on image width, 1 stride on image channel, 1 stride on image data set
    # "padding SAME" is zero padding
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
# max pooling function
def max_pool(x):
    # in case of ksize = [1, 3, 3, 1] and strides = [1, 2, 2, 1]
    # pick up pixel with max value from 3x3 region and stride 2 make the output of image size half.
    # finding the value from large area, so it make more robust against offset noise.
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')




########################################################################################################################
with tf.variable_scope('train_w'):
    # convolution layer 1
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable('conv1', [5, 5, 1, 32])
        b_conv1 = bias_variable('conv1', 32)
        # input x is transferred to x_image which is 4 dimension tensor [batch, in_height, in_width, in_channels]
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # W_conv1::4 dimension tensor [filter_height, filter_width, in_channels, channel_multiplier]
        # relu has sparse effect and make more faster learning
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # pooling layer 1
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool(h_conv1)

    # convolution layer 2
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable('conv2', [5, 5, 32, 64])
        b_conv2 = bias_variable('conv2', 64)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # pooling layer 2
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool(h_conv2)

    # fully connected layer
    with tf.name_scope('fully_connected') as scope:
        # 2 times Pooling, and the image size is 28->14->7
        # 64 number of filters
        n = 7 * 7 * 64
        # n = 7 * 7 * 64 (in), 1024 number of neuron (out)
        W_fc = weight_variable('fc', [n, 1024])
        b_fc = bias_variable('fc', 1024)
        h_pool2_flat = tf.reshape(h_pool2, [-1, n])
        # matmul is matrix A * matrix B
        h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

    # Drop out (to avoid over-fitting)
    with tf.name_scope('dropout') as scope:
        # dropout can remove the node which are not contribute the performance.
        # when training data was given, not using all nodes and randomly picking up the nodes and calculate the output,
        # in the backward process, not update all wight, only update the wight which are used for the above processing,
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

    # Fully Connected 2
    with tf.name_scope('fully_connected_for_PN') as scope:
        # output dimension: 15
        W_fc2 = weight_variable('fc2', [1024, bit_length])
        b_fc2 = bias_variable('fc2', bit_length)
        y_conv = 2.0*tf.nn.sigmoid(tf.matmul(h_fc_drop, W_fc2) + b_fc2) - 1.0


with tf.variable_scope('fixed_w'):
    with tf.name_scope('readout') as scope:
        y_conv2 = tf.nn.softmax(tf.matmul(y_conv, tf.cast(pn_, tf.float32)))
        # W_fc3 = weight_variable('fc3', [bit_length, 10])
        # y_conv2 = tf.nn.softmax(tf.matmul(y_conv, W_fc3))


########################################################################################################################

# loss function and training of the network
with tf.name_scope('loss') as scope:
    cross_entoropy = -tf.reduce_sum(y_org * tf.log(y_conv2))
    tf.summary.scalar('loss', cross_entoropy)

with tf.name_scope('training') as scope:
    # GradientDescentOptimizer
    # Adam algorithm
    optimizer = tf.train.AdamOptimizer(1e-4)
    optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="train_w")
    train_step = optimizer.minimize(cross_entoropy, var_list=optim_vars)

# evaluation of the network
with tf.name_scope('predict') as scope:
    # 1) threash-hold 0.5
    # corr_temp = tf.diag_part(tf.matmul(y_, tf.transpose(y_conv)))/bit_length
    # corr_bin = tf.greater(corr_temp, 0.5)
    # accuracy_step = tf.reduce_mean(tf.cast(corr_bin, tf.float32))

    # 2) find best candidate
    # y_est = tf.matmul(y_conv, tf.cast(pn_, tf.float32))
    # predict_step = tf.equal(tf.argmax(y_est, 1), tf.argmax(y_org, 1))
    # accuracy_step = tf.reduce_mean(tf.cast(predict_step, tf.float32))

    # 3) after multiply PN laywer
    # setting 1 to the second param, find the max row number in each column
    predict_step = tf.equal(tf.argmax(y_conv2, 1), tf.argmax(y_org, 1))
    # reduce_mean: calculate the mean value in all the array
    accuracy_step = tf.reduce_mean(tf.cast(predict_step, tf.float32))
    tf.summary.scalar('accuracy', accuracy_step)


# For display
summary = tf.summary.merge_all()


# setting of feed_dict (parameter)
def set_feed(images, labels, prob, pn):

    pncode = np.zeros((0, bit_length))
    for i in range(labels.shape[0]):

        if (labels[i,:] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode0.reshape(1,-1)]
        elif (labels[i,:] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode1.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode2.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode3.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode4.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode5.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 0, 1 , 0, 0, 0]).all():
            pncode = np.r_[pncode, pncode6.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all():
            pncode = np.r_[pncode, pncode7.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).all():
            pncode = np.r_[pncode, pncode8.reshape(1,-1)]
        elif (labels[i,:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
            pncode = np.r_[pncode, pncode9.reshape(1,-1)]

    # print(pncode.shape)
    return {x: images, y_: pncode, y_org : labels, keep_prob: prob, pn_: pn}

# start session
with tf.Session() as sess:

    # tf.Variable( ) need initialization
    sess.run(tf.global_variables_initializer())
    # for debug
    ## sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # for TensorBoard
    tw = tf.summary.FileWriter('log_dir', graph=sess.graph)

    # create the feed for test
    pn_all = np.c_[pncode0, pncode1, pncode2, pncode3, pncode4, pncode5, pncode6, pncode7, pncode8, pncode9]
    test_fd = set_feed(mnist.test.images, mnist.test.labels, 1, pn_all)


    # start training
    for step in range(10000):
        # 50 batch size
        # remember until which lines already were read,
        # and once finish to the end, it will be back to the start and repeat the reading again
        batch = mnist.train.next_batch(50)
        # batch[0] image, batch[1] label
        fd = set_feed(batch[0], batch[1], 0.5, pn_all)
        _, loss = sess.run([train_step, cross_entoropy], feed_dict=fd)
        # _, loss, temp = sess.run([train_step, cross_entoropy, W_fc3], feed_dict=fd)
        # print("xxxxx debug xxxxx", temp)

        if step % 100 == 0:
            acc, w_summary = sess.run([accuracy_step, summary], feed_dict=test_fd)
            print("step=", step, "loss=", loss, "acc=", acc)
            tw.add_summary(w_summary, step)


    # show the final result
    acc = sess.run(accuracy_step, feed_dict=test_fd)
    print("accuracy rate=", acc)




