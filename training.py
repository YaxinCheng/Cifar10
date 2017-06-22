import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
from datetime import datetime
import os

if not os.path.exists('tf_logs'): os.makedirs('tf_logs')
if not os.path.exists('model'): os.makedirs('model')

logdir = 'tf_logs/run-{}'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S'))

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
    shape = train_data.shape
    train_data = train_data.reshape((shape[0], 32, 32, 3))#.astype(np.float32)
    shape = valid_data.shape
    valid_data = valid_data.reshape((shape[0], 32, 32, 3)).astype(np.float32)
    shape = test_data.shape
    test_data = test_data.reshape((shape[0], 32, 32, 3)).astype(np.float32)
    train_labels, valid_labels, test_labels = np.array(train_labels), np.array(valid_labels), np.array(test_labels)
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

def fully_connected(X, neuron_number, name, activate_func=tf.nn.elu, dropout=True, keep_prob=0.5):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        with tf.variable_scope(name) as scope:
            try:
                weights = tf.get_variable('weight', shape=(n_inputs, neuron_number), initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(4 * 2 / (n_inputs + neuron_number))))
                biases = tf.get_variable('biases', shape=(neuron_number,), initializer=tf.constant_initializer(0))
            except ValueError: 
                scope.reuse_variables()
                weights = tf.get_variable('weight')
                biases = tf.get_variable('biases')
        result = tf.matmul(X, weights) + biases
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
    tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size,))
    tf_valid = tf.constant(valid_data)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test = tf.constant(test_data)
    tf_test_labels = tf.constant(test_labels) 
    
    with tf.name_scope('conv'):
        conv_weight_1 = tf.Variable(tf.truncated_normal((patch_size, patch_size, img_rgb, 16), stddev=0.1))
        conv_biases_1 = tf.Variable(tf.zeros(16)) 
        conv_weight_2 = tf.Variable(tf.truncated_normal((patch_size, patch_size, 16, 32), stddev=tf.sqrt(4 * 2 / (16 + 32))))
        conv_biases_2 = tf.Variable(tf.zeros(32))

    def model(input, dropout=True):
        with tf.name_scope('dnn'):
            input_conv_layer = tf.nn.elu(tf.nn.conv2d(input, conv_weight_1, [1, 2, 2, 1], padding='SAME') + conv_biases_1)
            #pool_layer = tf.nn.max_pool(input_conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            input_conv_layer = tf.nn.elu(tf.nn.conv2d(input_conv_layer, conv_weight_2, [1, 2, 2, 1], padding='SAME') + conv_biases_2) 
            #pool_layer = tf.nn.max_pool(input_conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            after_pooling_size = input_conv_layer.get_shape().as_list()
            fully_0 = tf.reshape(input_conv_layer, [after_pooling_size[0], after_pooling_size[1] * after_pooling_size[2] * after_pooling_size[3]])
            fully_1 = fully_connected(fully_0, 512, 'fully_1', dropout=dropout)
            fully_2 = fully_connected(fully_1, 256, 'fully_2', dropout=dropout) 
            fully_3 = fully_connected(fully_1, 128, 'fully_3', dropout=dropout) 
            fully_4 = fully_connected(fully_1, 64, 'fully_4', dropout=dropout) 
            logits = fully_connected(fully_4, num_labels, 'logits', activate_func=None, dropout=False)
            return logits

    with tf.name_scope('loss'):
        logits = model(tf_train)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) 
        v_logits = model(tf_valid)
        v_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_valid_labels, logits=v_logits)) 
        t_logits = model(tf_test)
        t_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_test_labels, logits=t_logits)) 

    with tf.name_scope('optimizer'):
        learning_rate = tf.placeholder(tf.float32, shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, tf_train_labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 
        correct = tf.nn.in_top_k(v_logits, tf_valid_labels, 1)
        v_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        correct = tf.nn.in_top_k(t_logits, tf_test_labels, 1)
        t_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope('visualization'):
        loss_s = tf.summary.scalar('Train_loss', loss)
        accu_s = tf.summary.scalar('Train_accu', accuracy)
        loss_v = tf.summary.scalar('Valid_loss', v_loss)
        accu_v = tf.summary.scalar('Valid_accu', v_accuracy)

    saver = tf.train.Saver()

epoches = 10
file_writer = tf.summary.FileWriter(logdir)
learning_r = 0.04

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for epoch in range(epoches):
        for batch in range(len(train_data) // batch_size):
            batch_data = get_batch_data(train_data, batch_size=batch_size, batch_num=batch)
            batch_labels = get_batch_data(train_labels, batch_size=batch_size, batch_num=batch)
            _, training_loss, training_accu = sess.run([optimizer, loss_s, accu_s], feed_dict={tf_train: batch_data, tf_train_labels: batch_labels, learning_rate: learning_r})
            if batch % 100 == 0: 
                step = epoch * batch_size + batch
                file_writer.add_summary(training_loss, step)
                file_writer.add_summary(training_accu, step)
                valid_l, valid_ac, valid_loss, valid_accu = sess.run([v_loss, v_accuracy, loss_v, accu_v])
                file_writer.add_summary(valid_loss, step)
                file_writer.add_summary(valid_accu, step)
                print('validation loss: ', valid_l, 'with accuracy:', valid_ac)
        valid_ac, valid_l = sess.run([v_accuracy, v_loss])
        print('After all batches: ')
        print('validation loss: ', valid_l, 'with accuracy:', valid_ac)
        #saver.save(sess, '/model/')
    print('\nTest case:')
    test_ac, test_l = sess.run([t_accuracy, t_loss])
    print('test loss:', test_l, 'with accuracy:', test_ac)
    #saver.save(sess, '/model/cifar10_{}.ckpt'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S')))
