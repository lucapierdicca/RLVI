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
            learning_rate=0.01,
            gamma=0.9,
            epsilon=1.0,
            memory_size=500,
            batch_size=32,
            hidden_units=256
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.hidden_units = hidden_units

        # experience replay memory
        self.memory = deque([], maxlen=memory_size)

        self.create_NNs()
        self.saver = tf.train.Saver() #var_list=tf.get_default_graph().get_collection('trainable_variables')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.copy_vars()


        # tf.summary.FileWriter("logs/", self.sess.graph)

        

    def create_NNs(self):
        # ------------------ Placeholders ------------------------
        self.s = tf.placeholder(tf.float32, [None, 2], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, 2], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  
        self.d = tf.placeholder(tf.float32, [None, ], name='d')  

               
        # ------------------ Conv Q ------------------
        with tf.variable_scope('Q'):
            fc1_e = tf.layers.dense(self.s, self.hidden_units, tf.nn.relu, trainable=True)
            self.q = tf.layers.dense(fc1_e, self.n_actions, trainable=True, name='q')


        # ------------------ Conv Q_tgt ------------------
        with tf.variable_scope('Q_tgt'):
            fc1_t = tf.layers.dense(self.s_, self.hidden_units, tf.nn.relu, trainable=False)
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

    def store_transition(self, s, a, r, s_, d):
        self.memory.append([s,a,r,s_,d])

    def get_action(self, s):

        # con Pr = epsilon scelgo random (uniformemente) tra tutte
        # con Pr = 1-epsilon scelgo greedy

        # get Q value for every action (array)
        actions_value = self.sess.run(self.q, feed_dict={self.s: [s]})
        max_Q = np.max(actions_value)
        
        if np.random.uniform() < self.epsilon:
            # choose a random action
            action = np.random.randint(0, self.n_actions)
        else:
            # choose the greedy action
            argmax_Q = np.argmax(actions_value)
            action = argmax_Q
            
        return action, max_Q

    
    def train(self,episode):

        # sample batch of transition from memory
        batch = random.sample(self.memory, self.batch_size)

        batch_s, batch_a, batch_r, batch_s_, batch_d = [],[],[],[],[]
        
        for transition in batch:

            batch_s.append(transition[0])
            batch_a.append(transition[1])
            batch_r.append(transition[2])
            batch_s_.append(transition[3])
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

        if episode % 10000 == 0:
            save_path = self.saver.save(self.sess, "./weights/weights.ckpt",
                 global_step=episode, write_meta_graph=False)

        return cost

    def eval():
        from pprint import pprint
        sess = tf.Session()
        #tf.reset_default_graph()
        saver = tf.train.import_meta_graph('./graph/graph.meta')
        saver.restore(sess, tf.train.latest_checkpoint("./weights/"))



        graph = tf.get_default_graph()
        s = graph.get_tensor_by_name('s:0')
        q = graph.get_tensor_by_name('Q/q/BiasAdd:0')

        states = []
        for i in range(7):
            for j in range(7):
                states.append([i,j])

        print(len(states))

        a = sess.run(q, {s:states})

        Q,r = {},0
        for i in range(7):
            for j in range(7):
                Q[str(i)+str(j)] = a[6*i+j]

        pprint(Q)




if __name__ == '__main__':
    pass
