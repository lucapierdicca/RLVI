import numpy as np
import tensorflow.compat.v1 as tf
from collections import deque
import random

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network
class DQN:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            gamma=0.9,
            epsilon=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.e_increment = e_increment
        self.e_max = epsilon
        self.epsilon = 0 if e_increment is not None else epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size


        # total learning step
        self.learn_step_counter = 0

        # experience replay memory
        self.memory = deque([], maxlen=memory_size)

        # consist of [target_net, evaluate_net]
        self.build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []


    def build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, 240, 240, 3], name='s')/255  # input State (batch, height, width, channel)
        self.s_ = tf.placeholder(tf.float32, [None, 240, 240, 3], name='s_')/255  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.d = tf.placeholder(tf.float32, [None, ], name='d')  # input Done

        #w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            conv1_e = tf.layers.conv2d(   # shape (84, 84, 3)
                inputs=self.s,
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

            self.q_eval = tf.layers.dense(fc1_e, self.n_actions)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            conv1_t = tf.layers.conv2d(   # shape (84, 84, 3)
                inputs=self.s_,
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

            self.q_next = tf.layers.dense(fc1_t, self.n_actions)


        with tf.variable_scope('q_target'):
            # reduce_max ritorna il massimo (wrt a) di q_next
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_') * (1-self.d)    # shape=(None, )
            #q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            # in questo modo ottengo il valore di q_eval dato lo stato e una specifica azione
            q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            # la media delle squared differenceS (una per ogni elemento del batch che gli dai in input)
            self.loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, 
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net'))


    def store_transition(self, s, a, r, s_, d):
        self.memory.append([s,a,r,s_,d])

    def choose_action(self, state, statelbl_to_img, id_to_orie):

        # recupero l'img dello stato [int,int,int] -> img
        state_img = statelbl_to_img[str(state[0])+str(state[1])+id_to_orie[state[2]]]
        # con Pr = epsilon scelgo la greedy
        # con Pr = 1-epsilon ne scelgo una random (uniformemente) tra tutte
        if np.random.uniform() < self.epsilon:
            # get Q value for every action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state_img]})
            # choose the greedy action
            action = np.argmax(actions_value)
        else:
            # choose a random action
            action = np.random.randint(0, self.n_actions)
        return action

    def train(self, statelbl_to_img, id_to_orie):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch of transition from memory
        batch = random.sample(self.memory, self.batch_size)

        batch_s, batch_a, batch_r, batch_s_, batch_d = [],[],[],[],[]

        for transition in batch:
            batch_s.append(statelbl_to_img[str(transition[0][0])+str(transition[0][1])+id_to_orie[transition[0][2]]])
            batch_a.append(transition[1])
            batch_r.append(transition[2])
            batch_s_.append(statelbl_to_img[str(transition[3][0])+str(transition[3][1])+id_to_orie[transition[3][2]]])
            batch_d.append(transition[4]) 
        

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.s: batch_s,
                self.a: batch_a,
                self.r: batch_r,
                self.s_: batch_s_,
                self.d: batch_d
            })


        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.e_increment if self.epsilon < self.e_max else self.e_max
        self.learn_step_counter += 1

        return cost

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training steps')
        plt.show()

if __name__ == '__main__':
    pass