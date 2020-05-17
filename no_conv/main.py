from environment import Grid
from network import DQN
from collections import deque
import pickle
from time import gmtime, strftime
import statistics as stat
import numpy as np



def train_loop(n_episode, offset_train, offset_copy, max_episode):

    global env

    tot_step_counter=0

    histories = {'episode_reward':[],
                 'argmax_Q':[],
                 'max_Q':[],
                 'epsilon':[],
                 'episode_len':[]}

    for episode in range(n_episode):

        episode_step_counter=0.0 
        episode_reward=0.0
        #episode_max_Q=0.0

        s = env.reset()
        _, max_Q, argmax_Q = agent.get_action(s)

        while True:

            a, _, _ = agent.get_action(s)

            s_, r, d = env.step(a)

            agent.store_transition(s, a, r, s_, d)

            ########################################

            s = list(s_)

            #episode_max_Q+=max_Q
            episode_reward+=r

            episode_step_counter+=1
            tot_step_counter += 1
            
            if d or episode_step_counter == max_episode:
                break


        print("episode: %d - goal: %d (%d) - e: %.4f (%d) - max_Q: %.5f - argmax_Q: %d - reward: %.1f" % 
            (episode+1, d, episode_step_counter,
                agent.epsilon, tot_step_counter, 
                max_Q, argmax_Q,  
                episode_reward))
        histories['max_Q'].append(max_Q)
        histories['argmax_Q'].append(argmax_Q)
        histories['episode_reward'].append(episode_reward)
        histories['epsilon'].append(agent.epsilon)
        histories['episode_len'].append(episode_step_counter)
        

        if (tot_step_counter > 10000) and (episode % offset_train == 0):
            batch = agent.sample()
            agent.train(batch)
            print('********* TRAIN ********')

        if (tot_step_counter > 10000) and (episode % offset_copy == 0):
            agent.copy_vars()
            print('********* COPY *********')

        if episode % 10000 == 0:
            print('********** SAVE *********')
            agent.saver.save(agent.sess, "./weights/weights.ckpt",
                 global_step=episode, write_meta_graph=False)
            

        #epsilon annealing
        if tot_step_counter > 20000:
            agent.epsilon = agent.epsilon - 0.9/110000


            
    pickle.dump(histories, open('histories.pickle','wb'))
    print('Game over')


def eval_loop(sess, Q_op, s_ph, n_episode, max_episode, epsilon):

    global env

    #evaluation
    policies = {'trained':[],'random':[]}
    
    for p in policies.keys():
        b_r = []
        b_sc = []
        


        for episode in range(n_episode):
            episode_reward=0
            episode_step_counter=0
            a_list = []
            
            s = env.reset()
            a_list.append(s)

            while True:
                if p == 'random': a = np.random.randint(0, env.n_actions)
                else:
                    if np.random.uniform() < epsilon:
                        a = np.random.randint(0, env.n_actions)
                    else:
                        actions_value = session.run(Q_op, feed_dict={s_ph: [s]})
                        a = np.argmax(actions_value)

                s_, r, d = env.step(a)
                s = list(s_)
                
                episode_reward+=r
                episode_step_counter+=1
                a_list.append(a)
                a_list.append(s)

                if d or episode_step_counter == max_episode:
                    break
            
            print("policy: %s - episode: %d - reward: %d" % (p,episode+1,episode_reward))
            print(a_list)
            b_r.append(episode_reward)
            b_sc.append(episode_step_counter)


        policies[p] = [stat.mean(b_r),
                       stat.variance(b_r),
                       stat.mean(b_sc),
                       stat.variance(b_sc)]

    return policies



#--------------------------------------------

TRAIN = True

# init environment
env = Grid()

if TRAIN:

    print("START TRAINING")

    # init training parameters
    n_episode = 50000
    offset_train = 1
    offset_copy = 300
    max_episode = 1000

    # agent init
    agent = DQN(env.n_actions,
                learning_rate=0.00001,#0.000001
                gamma=0.99,
                epsilon=1.0,
                memory_size=1000000,
                batch_size=64,
                hidden_units=128)
        
    # start training
    train_loop(n_episode, 
        offset_train, 
        offset_copy,
        max_episode)

else:

    print("START EVALUATION")

    n_episode = 10
    max_episode = 1000
    epsilon = 0.01

    # start evaluation
    session,Q_op,s_ph = DQN.restore()

    policy_score = eval_loop(session,Q_op,s_ph,
                             n_episode,
                             max_episode,
                             epsilon)

    print("Random")
    print("Avg R (%.2f - %.2f) - Avg length (%d - %.2f)" % (policy_score['random'][0],policy_score['random'][1],
                                                            policy_score['random'][2],policy_score['random'][3]))

    print("Trained")
    print("Avg R (%.2f - %.2f) - Avg length (%d - %.2f)" % (policy_score['trained'][0],policy_score['trained'][1],
                                                            policy_score['trained'][2],policy_score['trained'][3]))