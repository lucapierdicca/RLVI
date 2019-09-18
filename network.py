import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
from collections import deque
import random


# Deep Q Network
class DQN:
    def __init__(
            self,
            n_actions,
            n_history,
            learning_rate=0.01,
            gamma=0.9,
            epsilon=1.0,
            memory_size=500,
            batch_size=32,
            hidden_units=256
    ):
        self.n_actions = n_actions
        self.n_history = n_history
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.hidden_units = hidden_units


        # total learning step
        self.learn_step_counter = 0

        # experience replay memory
        self.memory = deque([], maxlen=memory_size)

        self.create_NNs()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.copy_vars()

        # tf.summary.FileWriter("logs/", self.sess.graph)

        




    def create_NNs(self):
        # ------------------ Placeholders ------------------------
        self.h = tf.placeholder(tf.float32, [None, 84, 84, 1*self.n_history], name='h')
        self.h_ = tf.placeholder(tf.float32, [None, 84, 84, 1*self.n_history], name='h_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  
        self.d = tf.placeholder(tf.float32, [None, ], name='d')  

        #w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        

        # ------------------ Conv Q ------------------
        with tf.variable_scope('Q'):
            conv1_e = tf.layers.conv2d(   # shape (240, 240, 3*n_history)
                inputs=self.h,
                filters=32,
                kernel_size=8,
                strides=4,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=contrib.layers.xavier_initializer(uniform=False),
                trainable=True
            )           # -> (20, 20, 32)

            conv2_e = tf.layers.conv2d(   # shape (20, 20, 32)
                inputs=conv1_e,
                filters=64,
                kernel_size=4,
                strides=2,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=contrib.layers.xavier_initializer(uniform=False),
                trainable=True
            )           # -> (9, 9, 64)

            conv3_e = tf.layers.conv2d(   # shape (9, 9, 64)
                inputs=conv2_e,
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=contrib.layers.xavier_initializer(uniform=False),
                trainable=True
            )           # -> (7, 7, 64)

            flat_e = tf.layers.flatten(conv3_e, data_format='channels_last')
            fc1_e = tf.layers.dense(flat_e, self.hidden_units, tf.nn.relu, trainable=True)
            self.q = tf.layers.dense(fc1_e, self.n_actions, trainable=True)

        

        # ------------------ Conv Q_tgt ------------------
        with tf.variable_scope('Q_tgt'):
            conv1_t = tf.layers.conv2d(   # shape (240, 240, 3*n_history)
                inputs=self.h_,
                filters=32,
                kernel_size=8,
                strides=4,
                padding='same',
                activation=tf.nn.relu,
                trainable=False
            )           # -> (20, 20, 32)

            conv2_t = tf.layers.conv2d(   # shape (20, 20, 32)
                inputs=conv1_t,
                filters=64,
                kernel_size=4,
                strides=2,
                padding='same',
                activation=tf.nn.relu,
                trainable=False
            )           # -> (9, 9, 64)

            conv3_t = tf.layers.conv2d(   # shape (9, 9, 64)
                inputs=conv2_t,
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                trainable=False
            )           # -> (7, 7, 64)

            flat_t = tf.layers.flatten(conv3_t, data_format='channels_last')
            fc1_t = tf.layers.dense(flat_t, self.hidden_units, tf.nn.relu, trainable=False)
            self.q_tgt = tf.layers.dense(fc1_t, self.n_actions, trainable=False)


            # --------------- Copying -------------------
            self.copy_vars_op = [tf.assign(tgt, nor) for tgt, nor in zip(tf.trainable_variables('Q_tgt'), 
                tf.trainable_variables('Q'))]


            # ---------------- Training ---------------------
            # reduce_max ritorna il massimo (wrt a) di q_tgt
            target = self.r + (self.gamma*tf.reduce_max(self.q_tgt, axis=1)*(1-self.d))
    
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            # in questo modo ottengo il valore di q dato lo stato e una specifica azione
            q_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)    # shape=(None, )
            # la media delle squared differenceS (una per ogni elemento del batch che gli dai in input)
            self.loss = tf.reduce_mean(tf.squared_difference(target, q_wrt_a))
            
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


    def copy_vars(self):
        self.sess.run(self.copy_vars_op)

    def store_transition(self, h, a, r, h_, d):
        self.memory.append([h,a,r,h_,d])

    def get_action(self, h, statelbl_to_img, id_to_orie):

        #print("--------------->%f" % self.epsilon)


        # dalla history of state lbl a history of state img = stacked input images
        history_img = [] 
        for state in h:
            history_img.append(statelbl_to_img[str(state[0])+str(state[1])+id_to_orie[state[2]]])
        
        if len(history_img) > 1:
            history_img = [np.dstack(tuple(history_img))]
        

        # con Pr = epsilon scelgo random (uniformemente) tra tutte
        # con Pr = 1-epsilon ne scelgo greedy

        # get Q value for every action
        actions_value = self.sess.run(self.q, feed_dict={self.h: history_img})
        max_Q = np.max(actions_value)
        
        if np.random.uniform() < self.epsilon:
            # choose a random action
            action = np.random.randint(0, self.n_actions)
        else:
            # choose the greedy action
            action = np.argmax(actions_value)
            
        return action, max_Q

    
    def train(self, statelbl_to_img, id_to_orie):

        # sample batch of transition from memory
        batch = random.sample(self.memory, self.batch_size)

        batch_h, batch_a, batch_r, batch_h_, batch_d = [],[],[],[],[]
        
        for transition in batch:
            
            history_img = []
            for state in transition[0]:
                history_img.append(statelbl_to_img[str(state[0])+str(state[1])+id_to_orie[state[2]]])
                
            if len(history_img) == 1:
                history_img = history_img[0]
            else:
                history_img = np.dstack(tuple(history_img))

            batch_h.append(history_img)
            
            batch_a.append(transition[1])
            batch_r.append(transition[2])
            
            history_img = []
            for state in transition[3]:
                history_img.append(statelbl_to_img[str(state[0])+str(state[1])+id_to_orie[state[2]]])

            if len(history_img) == 1:
                history_img = history_img[0]
            else:
                history_img = np.dstack(tuple(history_img))

            batch_h_.append(history_img)
            
            batch_d.append(transition[4]) 
        

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.h: batch_h,
                self.a: batch_a,
                self.r: batch_r,
                self.h_: batch_h_,
                self.d: batch_d
            })


        # annealing epsilon
        self.epsilon = max(((-0.9/200000.0)*self.learn_step_counter) + 1.0, 0.1)
        self.learn_step_counter+=1

        return cost



if __name__ == '__main__':
    pass
