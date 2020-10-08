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


        # experience replay memory
        self.memory = deque([], maxlen=memory_size)

        self.create_NNs()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.copy_vars()

        tf.summary.FileWriter("logs/", self.sess.graph)


    def create_NNs(self):
        # ------------------ Placeholders ------------------------
        self.h = tf.placeholder(tf.float32, [None, 40, 40, 1*self.n_history], name='h')
        self.h_ = tf.placeholder(tf.float32, [None, 40, 40, 1*self.n_history], name='h_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  
        self.d = tf.placeholder(tf.float32, [None, ], name='d')  

        

        # ------------------ Conv Q ------------------
        with tf.variable_scope('Q'):
            conv1_e = tf.layers.conv2d(   
                inputs=self.h,
                filters=8,
                kernel_size=19,
                strides=2,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=contrib.layers.xavier_initializer(uniform=False),
                trainable=True
            )           

            conv2_e = tf.layers.conv2d(  
                inputs=conv1_e,
                filters=16,
                kernel_size=9,
                strides=2,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=contrib.layers.xavier_initializer(uniform=False),
                trainable=True
            )           

  

            flat_e = tf.layers.flatten(conv2_e, data_format='channels_last')
            fc1_e = tf.layers.dense(flat_e, self.hidden_units, tf.nn.relu, trainable=True)
            self.q = tf.layers.dense(fc1_e, self.n_actions, trainable=True, name='q')

        

        # ------------------ Conv Q_tgt ------------------
        with tf.variable_scope('Q_tgt'):
            conv1_t = tf.layers.conv2d(  
                inputs=self.h_,
                filters=8,
                kernel_size=19,
                strides=2,
                padding='same',
                activation=tf.nn.relu,
                trainable=False
            )          

            conv2_t = tf.layers.conv2d(   
                inputs=conv1_t,
                filters=16,
                kernel_size=9,
                strides=2,
                padding='same',
                activation=tf.nn.relu,
                trainable=False
            )           


            flat_t = tf.layers.flatten(conv2_t, data_format='channels_last')
            fc1_t = tf.layers.dense(flat_t, self.hidden_units, tf.nn.relu, trainable=False)
            self.q_tgt = tf.layers.dense(fc1_t, self.n_actions, trainable=False)


        # --------------- Copying -------------------
        self.copy_vars_op = [tf.assign(tgt, nor) for tgt, nor in zip(tf.trainable_variables('Q_tgt'), 
            tf.trainable_variables('Q'))]


        # ---------------- Training ---------------------
        # reduce_max ritorna il massimo (wrt a) di q_tgt
        target = self.r + (self.gamma*tf.reduce_max(self.q_tgt, axis=1)*(1-self.d))
        #target = tf.stop_gradient(target)
        a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
        # in questo modo ottengo il valore di q dato lo stato e una specifica azione
        q_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)    # shape=(None, )
        # la media delle squared differenceS (una per ogni elemento del batch che gli dai in input)
        self.loss = tf.reduce_mean(tf.squared_difference(target, q_wrt_a))
        
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # export only Q graph (not Q_tgt)
        tf.train.export_meta_graph(filename='./graph/graph.meta')


    def copy_vars(self):
        self.sess.run(self.copy_vars_op)

    def store_transition(self, h, a, r, h_, d):
        self.memory.append([h,a,r,h_,d])

    def get_action(self, h, statelbl_to_img, id_to_orie):

        #print("--------------->%f" % self.epsilon)
        # dalla history of state lbl a history of state img = stacked input images
        history_img = [] 
        for state in h:
            history_img.append(statelbl_to_img[str(state[0])+str(state[1])+id_to_orie[state[2]]][0])
        
        if len(history_img) > 1:
            history_img = [np.dstack(tuple(history_img))]
        

        # con Pr = epsilon scelgo random (uniformemente) tra tutte
        # con Pr = 1-epsilon scelgo greedy

        # get Q value for every action
        actions_value = self.sess.run(self.q, feed_dict={self.h: history_img})
        max_Q = actions_value[0][0] #f
        argmax_Q = np.argmax(actions_value)
        
        if np.random.uniform() < self.epsilon:
            # choose a random action
            action = np.random.randint(0, self.n_actions)
        else:
            # choose the greedy action
            argmax_Q = np.argmax(actions_value)
            action = argmax_Q
            
        return action, max_Q, argmax_Q

    
    def train(self, statelbl_to_img, id_to_orie):

        # sample batch of transition from memory
        batch = random.sample(self.memory, self.batch_size)

        batch_h, batch_a, batch_r, batch_h_, batch_d = [],[],[],[],[]
        
        for transition in batch:
            
            history_img = []
            for state in transition[0]:
                history_img.append(statelbl_to_img[str(state[0])+str(state[1])+id_to_orie[state[2]]][0])
                
            if len(history_img) == 1:
                history_img = history_img[0]
            else:
                history_img = np.dstack(tuple(history_img))

            batch_h.append(history_img)
            
            batch_a.append(transition[1])
            batch_r.append(transition[2])
            
            history_img = []
            for state in transition[3]:
                history_img.append(statelbl_to_img[str(state[0])+str(state[1])+id_to_orie[state[2]]][0])

            if len(history_img) == 1:
                history_img = history_img[0]
            else:
                history_img = np.dstack(tuple(history_img))

            batch_h_.append(history_img)
            
            batch_d.append(transition[4]) 
        

        self.sess.run(
            self.train_op,
            feed_dict={
                self.h: batch_h,
                self.a: batch_a,
                self.r: batch_r,
                self.h_: batch_h_,
                self.d: batch_d
            })



    @staticmethod
    def restore():
        # restore the last weights and the graph
        sess = tf.Session()
        saver = tf.train.import_meta_graph('./paraboloid_reward_h/graph/graph.meta')
        
        saver.restore(sess, tf.train.latest_checkpoint("./paraboloid_reward_h/weights"))

        graph = tf.get_default_graph()
        h = graph.get_tensor_by_name('h:0')
        q = graph.get_tensor_by_name('Q/q/BiasAdd:0')

        return sess, q, h





if __name__ == '__main__':
    pass


