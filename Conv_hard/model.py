import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np


n_features = 5
gamma = 0.99
n_actions = 3
lr = 0.01


# ------------------ all inputs ------------------------
s = tf.placeholder(tf.float32, [None, 84, 84, 1], name='s')# input State (batch, height, width, channel)
s_ = tf.placeholder(tf.float32, [None, 84, 84, 1], name='s_')# input Next State
r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

#w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

# ------------------ build evaluate_net ------------------
with tf.variable_scope('eval_net'):
    conv1_e = tf.layers.conv2d(   # shape (84, 84, 3)
        inputs=s,
        filters=32,
        kernel_size=8,
        strides=4,
        padding='same',
        activation=tf.nn.relu
    )           # -> (20, 20, 32)

    conv2_e = tf.layers.conv2d(   # shape (20, 20, 32)
        inputs=conv1_e,
        filters=64,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )           # -> (9, 9, 64)

    conv3_e = tf.layers.conv2d(   # shape (9, 9, 64)
        inputs=conv2_e,
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )           # -> (7, 7, 64)

    flat_e = tf.layers.flatten(conv3_e, data_format='channels_last')

    fc1_e = tf.layers.dense(flat_e, 512, tf.nn.relu)

    q_eval = tf.layers.dense(fc1_e, n_actions)

# ------------------ build target_net ------------------
with tf.variable_scope('target_net'):
    conv1_t = tf.layers.conv2d(   # shape (84, 84, 3)
        inputs=s_,
        filters=32,
        kernel_size=8,
        strides=4,
        padding='same',
        activation=tf.nn.relu
    )           # -> (20, 20, 32)

    conv2_t = tf.layers.conv2d(   # shape (20, 20, 32)
        inputs=conv1_t,
        filters=64,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )           # -> (9, 9, 64)

    conv3_t = tf.layers.conv2d(   # shape (9, 9, 64)
        inputs=conv2_t,
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )           # -> (7, 7, 64)

    flat_t = tf.layers.flatten(conv3_t, data_format='channels_last')

    fc1_t = tf.layers.dense(flat_t, 512, tf.nn.relu)

    q_next = tf.layers.dense(fc1_t, n_actions)


with tf.variable_scope('q_target'):
    q_target = r + gamma * tf.reduce_max(q_next, axis=1, name='Qmax_s_')    # shape=(None, )
    #q_target = tf.stop_gradient(q_target)
with tf.variable_scope('q_eval'):
    a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
    q_eval_wrt_a = tf.gather_nd(params=q_eval, indices=a_indices)    # shape=(None, )
with tf.variable_scope('loss'):
    # la media delle squared differenceS (una per ogni elemento del batch che gli dai in input)
    loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_wrt_a, name='TD_error'))
with tf.variable_scope('train'):
    train_op = tf.train.RMSPropOptimizer(lr).minimize(loss, 
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net'))




sess = tf.Session()

sess.run(tf.global_variables_initializer())


import cv2

# Load an color image in grayscale
img = cv2.imread("./env_renders/00N.png",cv2.IMREAD_GRAYSCALE)
print(img.shape)


q_eval_res, q_eval_wrt_a_res  = sess.run([q_eval,q_eval_wrt_a], feed_dict={s:[img, img, img], a:[0,2,1]})


from pprint import pprint

pprint(q_eval_res)
pprint(q_eval_wrt_a_res)

# # $ tensorboard --logdir=logs/ --host localhost --port 8088
# tf.summary.FileWriter("logs/", sess.graph)
# from pprint import pprint


# pprint(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net'))
# pprint(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net'))

# pprint(tf.trainable_variables('eval_net'))
