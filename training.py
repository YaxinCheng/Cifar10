import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
from datetime import datetime
import os
import sys

if not os.path.exists('tf_logs'): os.makedirs('tf_logs')
if not os.path.exists('model'): os.makedirs('model')

logdir = 'tf_logs/run-{}'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S'))
img_width, img_length, img_rgb = 32, 32, 3

def loadData():
    fileName = 'data/data_batch_{index}' 
    for index in range(1, 6):
        with open(fileName.format(index=index), 'rb') as file:
            rawData = pickle.load(file, encoding='bytes')
            if index == 1:
                train_data, train_labels = rawData[b'data'], rawData[b'labels']
            elif index == 5:
                valid_data, valid_labels = rawData[b'data'], rawData[b'labels']
            else:
                train_data = np.concatenate([train_data, rawData[b'data']]) 
                train_labels += rawData[b'labels']
    with open('data/test_batch', 'rb') as file:
        rawData = pickle.load(file, encoding='bytes')
        test_data, test_labels = rawData[b'data'], rawData[b'labels']
    global img_width, img_length, img_rgb
    newshape = (-1, img_length, img_width, img_rgb)
    train_data = train_data.reshape(newshape).astype(np.float32)
    valid_data = valid_data.reshape(newshape).astype(np.float32)
    test_data = test_data.reshape(newshape).astype(np.float32)
    train_labels, valid_labels, test_labels = np.array(train_labels), np.array(valid_labels), np.array(test_labels)
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

def conv_connected(X, filter_shape, feature_maps, strides, name, padding='SAME', activate_func=tf.nn.elu, pool_func=tf.nn.max_pool, ksize=None, pstrides=None, ppadding='SAME', normalize_func=tf.nn.lrn):
    input_shape = int(X.get_shape()[-1])
    filter_shape = (filter_shape,) * 2 + (input_shape, feature_maps,)
    strides = (1,) + (strides,) * 2 + (1,)
    with tf.name_scope(name):
        with tf.variable_scope(name) as scope:
            try:
                filter = tf.get_variable('weight', shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                biases = tf.get_variable('biases', shape=(filter_shape[-1],), initializer=tf.constant_initializer(0))
            except ValueError:
                scope.reuse_variables()
                filter = tf.get_variable('weight')
                biases = tf.get_variable('biases')
        result = tf.nn.conv2d(X, filter, strides, padding) + biases
        if normalize_func: conv = normalize_func(result)
        if activate_func: result = activate_func(result)
        if pool_func and ksize and pstrides and ppadding:
            ksize = (1,) + (ksize,) * 2 + (1,)
            pstrides = (1,) + (pstrides,) * 2 + (1,)
            result = pool_func(result, ksize, pstrides, ppadding)
        return result

def fully_connected(X, neuron_number, name, batch_norm=True, activate_func=tf.nn.elu, dropout=False, keep_prob=0.5):
    n_inputs = int(X.get_shape()[1])
    if batch_norm:
        with tf.name_scope(name):
            mean, variance = tf.nn.moments(X, axes=0)
            with tf.variable_scope(name) as scope:
                try:
                    offset = tf.get_variable('offset', shape=(n_inputs,), initializer=tf.truncated_normal_initializer(stddev=0.1))
                    scale = tf.get_variable('scale', shape=(n_inputs,), initializer=tf.truncated_normal_initializer(stddev=0.1))
                except:
                    scope.reuse_variables()
                    offset = tf.get_variable('offset')
                    scale = tf.get_variable('scale')
        X = tf.nn.batch_normalization(X, mean, variance, offset, scale, 0.0001)
    with tf.name_scope(name):
        with tf.variable_scope(name) as scope:
            try:
                weights = tf.get_variable('weight', shape=(n_inputs, neuron_number), initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(4 * 2 / (n_inputs + neuron_number)))) 
                biases = tf.get_variable('biases', shape=(neuron_number,), initializer=tf.constant_initializer(0))
            except ValueError: 
                scope.reuse_variables()
                weights = tf.get_variable('weight')
                biases = tf.get_variable('biases')
        if not dropout: weights = weights * keep_prob# Weights scaling
        result = tf.matmul(X, weights) + biases
        if activate_func: result = activate_func(result)
        if dropout: result = tf.nn.dropout(result, keep_prob=keep_prob)
        else: result = tf.nn.dropout(result, 1)
        return result

def get_batch_data(data, batch_num, batch_size):
    lowerbound = (batch_num * batch_size) % len(data)
    if lowerbound + batch_size < len(data):
        upperbound = lowerbound + batch_size 
        return data[lowerbound:upperbound]
    else:
        first = data[lowerbound:]
        second = data[:(batch_size  + lowerbound) % len(data)]
        return np.concatenate((first, second))

batch_size = 64
num_labels = 10 
patch_size = 3

train_data, train_labels, valid_data, valid_labels, test_data, test_labels = loadData()

graph = tf.Graph()
with graph.as_default():
    tf_train = tf.placeholder(tf.float32, shape=(batch_size, img_width, img_length, img_rgb))
    tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size,))
    tf_valid = tf.constant(valid_data)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test = tf.constant(test_data)
    tf_test_labels = tf.constant(test_labels) 

    def model(input, dropout=True):
        with tf.name_scope('dnn'):
            conv = conv_connected(input, 1, 32, strides=1, name='conv0_p', activate_func=None, normalize_func=None)
            conv = conv_connected(conv, patch_size, 32, strides=1, name='conv0', ksize=3, pstrides=2)
            conv = conv_connected(conv, 1, 64, strides=1, name='conv1_p', activate_func=None, normalize_func=None)
            conv = conv_connected(conv, patch_size, 64, strides=1, name='conv1', ksize=3, pstrides=2)
            conv = conv_connected(conv, 1, 128, strides=1, name='conv2_p', activate_func=None, normalize_func=None)
            conv = conv_connected(conv, patch_size, 128, strides=1, name='conv2', ksize=3, pstrides=2)
            conv = conv_connected(conv, 1, 256, strides=1, name='conv3_p', activate_func=None, normalize_func=None)
            conv = conv_connected(conv, patch_size, 256, strides=1,  name='conv3', ksize=3, pstrides=2)
            shape = conv.get_shape().as_list()
            fully = tf.reshape(conv, [shape[0], shape[1] * shape[2] * shape[3]])
            fully = fully_connected(fully, 1024, name='layer0', dropout=dropout, keep_prob=0.3)
            fully = fully_connected(fully, 512, name='layer1', dropout=dropout, keep_prob=0.3)
            fully = fully_connected(fully, 256, name='layer2', dropout=dropout, keep_prob=0.4)
            fully = fully_connected(fully, 128, name='layer3', dropout=dropout)
            fully = fully_connected(fully, 80, name='layer4', dropout=dropout) 
            logits = fully_connected(fully, 10, 'logits', activate_func=None, dropout=False, keep_prob=1)
            return logits

    def accuracy(logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope('loss'):
        logits = model(tf_train, dropout=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) 
        v_logits = model(tf_valid, dropout=False)
        v_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_valid_labels, logits=v_logits)) 
        t_logits = model(tf_test, dropout=False)
        t_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_test_labels, logits=t_logits)) 

    with tf.name_scope('optimizer'):
        learning_rate = tf.placeholder(tf.float32, shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('eval'):
        tr_accuracy = accuracy(logits, tf_train_labels) 
        v_accuracy = accuracy(v_logits, tf_valid_labels)
        t_accuracy = accuracy(t_logits, tf_test_labels)

    with tf.name_scope('visualization'):
        loss_s = tf.summary.scalar('Train_loss', loss)
        loss_v = tf.summary.scalar('Valid_loss', v_loss)

    saver = tf.train.Saver()

epoches = 9001
file_writer = tf.summary.FileWriter(logdir)
learning_r = 0.0007#0.0007 was great

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for epoch in range(epoches):
        try:
            batch_data = get_batch_data(train_data, batch_size=batch_size, batch_num=epoch)
            batch_labels = get_batch_data(train_labels, batch_size=batch_size, batch_num=epoch)
            _, training_loss = sess.run([optimizer, loss_s], feed_dict={tf_train: batch_data, tf_train_labels: batch_labels, learning_rate: learning_r})
            if epoch % 500 == 0: 
                file_writer.add_summary(training_loss, epoch)
                valid_l, valid_loss, v_accu = sess.run([v_loss, loss_v, v_accuracy])
                print(epoch, 'Valid loss', valid_l, 'Accuracy', '{}%'.format(v_accu * 100))
                file_writer.add_summary(valid_loss, epoch)
                file_writer.add_summary(training_loss, epoch)
        except KeyboardInterrupt:
            sys.exit(0)
        #saver.save(sess, 'model/cifar10_partly.ckpt')
    print('\nTest case:')
    test_ac, test_l = sess.run([t_accuracy, t_loss])
    print('test loss:', test_l, 'with accuracy: {}%\n'.format(test_ac * 100))
    #saver.save(sess, 'model/cifar10.ckpt')
