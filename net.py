#coding=utf-8

import tensorflow as tf

dim = 16

def flow_net(img1, img2, reuse=True, training=True):
    conv_i1 = img_pro(img1, reuse)
    conv_i2 = img_pro(img2, reuse)
    with tf.variable_scope("img_cmb"):
        conv1 = tf.layers.conv2d(conv_i1 + conv_i2, dim * 8, [3, 3], 2, padding='same',
                                 activation=tf.nn.relu, name='conv1')
        reconv1 = tf.layers.conv2d_transpose(conv1, dim * 4, [3, 3], 2, padding='same',
                                             activation=tf.nn.relu, name='reconv1')
        reconv2 = tf.layers.conv2d_transpose(reconv1, dim * 2, [3, 3], 2, padding='same',
                                             activation=tf.nn.relu, name='reconv2')
        reconv3 = tf.layers.conv2d_transpose(reconv2, dim, [3, 3], 2, padding='same',
                                             activation=tf.nn.relu, name='reconv3')
        conv2 = tf.layers.conv2d(reconv3, 2, [3, 3], 1, padding='same',
                                 activation=None, name='conv2')
        return conv2

def img_pro(img, reuse=True):
    with tf.variable_scope("img_pro", reuse):
        conv1 = tf.layers.conv2d(img, dim, [3,3], 2, padding='same', activation=tf.nn.relu,
                                 name='conv1')
        conv2 = tf.layers.conv2d(conv1, dim * 2, [3, 3], 2, padding='same', activation=tf.nn.relu,
                                 name='conv2')
        conv3 = tf.layers.conv2d(conv2, dim * 4, [3, 3], 2, padding='same', activation=tf.nn.relu,
                                 name='conv3')
        conv4 = tf.layers.conv2d(conv3, dim * 8, [3, 3], 1, padding='same', activation=tf.nn.relu,
                                 name='conv4')
        return conv4

def img_dis(img1, reuse=True, training=True):
    conv_i1 = img_pro(img1, reuse)
    with tf.variable_scope("img_diff"):
        conv1 = tf.layers.conv2d(conv_i1, dim * 8, [3, 3], 2, padding='same',
                                 activation=tf.nn.relu, name='conv1')
        fl1 = tf.layers.flatten(conv1, name='fl1')
        dense1 = tf.layers.dense(fl1, dim * dim * dim, activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(dense1, dim * dim, activation=tf.nn.relu, name='dense2')
        diff = tf.layers.dense(dense2, 1, activation=tf.nn.relu, name='dense3')
        return diff