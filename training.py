import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np

def loadData():
    fileName = 'data/data_batch_{index}' 
    for index in range(1, 6):
        with open(fileName.format(index=index), 'rb') as file:
            rawData = pickle.load(file, encoding='bytes')
            if index == 0:
                train_data, train_labels = rawData[b'data'], rawData[b'labels']
            elif index == 5:
                valid_data, valid_labels = rawData[b'data'], rawData[b'labels']
            else:
                train_data = np.concatenate([train_data, rawData[b'data']]) 
                train_labels += rawData[b'labels']
    with open('data/test_batch', 'rb') as file:
        rawData = pickle.load(file, encoding='bytes')
        test_data, test_labels = rawData[b'data'], rawData[b'labels']
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

def fully_connected(X, neuron_number, name, activate_func=tf.nn.elu, dropout=True, keep_prob=0.5):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        weights = tf.Variable(tf.truncate_normal((n_inputs, neuron_number), stddev=tf.sqrt(4 * 2 / (n_inputs + neuron_number)), name='weights'))
        biases = tf.Variable(tf.zeros([neuron_number]), name='biases')
        result = tf.mat_mul(X, weights) + biases
        if activate_func: result = activate_func(result)
        if dropout: result = tf.nn.dropout(result, keep_prob=keep_prob)
        return result

def get_batch_data(data, batch_num, batch_size):
    lowerbound = (batch_num * batch_size) % len(data)
    upperbound = lowerbound + batch_size 
    return data[lowerbound:upperbound]

batch_size = 128
img_width, img_length = 32, 32
img_rgb = 3
num_labels = 10

patch_size = 4

train_data, train_labels, valid_data, valid_labels, test_data, test_labels = loadData()

graph = tf.Graph()
with graph.as_default():
    tf_train = tf.placeholder(tf.float32, shape=(batch_size, img_width, img_length, img_rgb))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,))
    running_mode = tf.placeholder(tf.string, shape=())
    dropoutCond = tf.equal(running_mode, tf.constant('train'))
    
    with tf.name_scope('conv'):
        conv_weight_1 = tf.Variable(tf.truncate_normal((patch_size, patch_size, img_rgb, 16), stddev=0.1))
        conv_biases_1 = tf.Variable(tf.zeros(16)) 
        conv_weight_2 = tf.Variable(tf.truncate_normal((patch_size, patch_size, 16, 32), stddev=tf.sqrt(4 *2 / (16 + 32))))
        conv_biases_2 = tf.Variable(tf.zeros(32))

    with tf.name_scope('dnn'):
        input_conv_layer = tf.nn.elu(tf.nn.conv2d(tf_train, conv_weight_1, [1, 1, 1, 1], padding='SAME'))
        pool_layer = tf.nn.max_pool(input_conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        input_conv_layer = tf.nn.elu(tf.nn.conv2d(pool_layer, conv_weight_2, [1, 1, 1, 1], padding='SAME'))
        pool_layer = tf.nn.max_pool(input_conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        after_pooling_size = pool_layer.get_shape()
        input_for_fully = tf.reshape(pool_layer, shape=(after_pooling_size[0], after_pooling_size[1] * after_pooling_size[2] * after_pooling_size[3]))
        fully_1 = tf.where(dropoutCond, fully_connected(input_for_fully, 512, 'fully_1'), fully_connected(input_for_fully, 512, 'fully_1', dropout=False))
        fully_2 = tf.where(dropoutCond, fully_connected(fully_1, 256, 'fully_2'), fully_connected(fully_1, 256, 'fully_2', dropout=False)) 
        logits = fully_connected(fully_2, num_labels, 'logits', activate_func=None, dropout=False)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) 

    with tf.name_scope('optimizer'):
        learning_rate = tf.placeholder(tf.float32, shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, tf_train_labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 

    with tf.name_scopre('visualization'):
        loss_s = tf.summary.scalar(running_mode + '_Loss', loss)
        accu_s = tf.summary.scalar(running_mode + '_Accu', accuracy)

epoches = 1001
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in epoches:
        for batch in range(train_data.shape() // batch_size):
            batch_data = get_batch_data(train_data, batch_size=128, batch_num=batch)
            batch_labels = get_batch_data(train_labels, batch_size=128, batch_num=batch)
            _, ac, l = sess.run([optimizer, accuracy, loss], feed_dict={tf_train: batch_data, tf_train_labels: batch_labels, learning_rate: 0.01, running_mode: 'train'})
            # Add ac to the tensorboard
        valid_ac, valid_l = sess.run([accuracy, loss], feed_dict={tf_train: valid_data, tf_train_labels: valid_data_labels, running_mode:'valid'})
        # Add valid accu to the tensorboard
    test_ac, test_l = sess.run([accuracy, loss], feed_dict={tf_train: test_data, tf_train_labels: test_data_labels, running_mode: 'test'})
    # Add the test accu to the tensorboard


