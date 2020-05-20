from environment import Grid
from network import DQN
from collections import deque
import pickle
from time import gmtime, strftime
import numpy as np
import statistics as stat


def reset_history():

    global history
    for i in range(history.maxlen):
        history.append(['_','_',0])


def train_loop(n_episode, offset_train, offset_copy, max_episode):

    global env, history
    statelbl_to_img, id_to_orie = env.get_renders()
    action_space = {'f':0, 'tl':1, 'tr':2}
    id_to_action = {v:k for k,v in action_space.items()}

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


        reset_history()
        s = env.reset()
        history.append(s)
        h = list(history)

        _, max_Q, argmax_Q = agent.get_action(h, statelbl_to_img, id_to_orie)

        while True:
            
            a, _,_ = agent.get_action(h, statelbl_to_img, id_to_orie)

            s_, r, d = env.step(a)
            history.append(s_)
            h_ = list(history)

            # a transition is [[history],int,int,[history_],int]
            agent.store_transition(h, a, r, h_, d)
            #print(h,a,h_)

            ##########################################

            h = list(h_)

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
            print('********* TRAIN ********')
            agent.train(statelbl_to_img, id_to_orie)


        if (tot_step_counter > 10000) and (episode % offset_copy == 0):
            print('********* COPY *********')
            agent.copy_vars()


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

    global env, history
    statelbl_to_img, id_to_orie = env.get_renders()

    #evaluation
    policies = {'trained':[],'random':[]}
    
    for p in policies.keys():
        b_r = []
        b_sc = []

        for episode in range(n_episode):
            episode_reward=0
            episode_step_counter=0
            
            reset_history()
            s = env.reset()
            history.append(s)

            while True:
                if p == 'random': a = np.random.randint(0, env.n_actions)
                else:
                    if np.random.uniform() < epsilon:
                        a = np.random.randint(0, env.n_actions)
                    else:
                        history_img = [] 
                        for state in history:
                            history_img.append(statelbl_to_img[str(state[0])+str(state[1])+id_to_orie[state[2]]][0])
                        
                        if len(history_img) > 1:
                            history_img = [np.dstack(tuple(history_img))]

                        actions_value = session.run(Q_op, feed_dict={h_ph: history_img})
                        a = np.argmax(actions_value)

                s_, r, d = env.step(a)
                history.append(s_)
                
                episode_reward+=r
                episode_step_counter+=1

                if d or episode_step_counter == max_episode:
                    break
            
            print("policy: %s - episode: %d - reward: %d" % (p,episode+1,episode_reward))
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

# init history 
n_history = 4
history = deque([], maxlen=n_history)

if TRAIN:

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

    n_episode = 100
    max_episode = 1000
    epsilon = 0.05

    # start evaluation
    session,Q_op,h_ph = DQN.restore()

    policy_score = eval_loop(session,Q_op,h_ph,
                             n_episode,
                             max_episode,
                             epsilon)

    print("Random")
    print("Avg R (%.2f - %.2f) - Avg length (%d - %.2f)" % (policy_score['random'][0],policy_score['random'][1],
                                                            policy_score['random'][2],policy_score['random'][3]))

    print("Trained")
    print("Avg R (%.2f - %.2f) - Avg length (%d - %.2f)" % (policy_score['trained'][0],policy_score['trained'][1],
                                                            policy_score['trained'][2],policy_score['trained'][3]))


# Plain Reward
# Random
# Avg R (-1309.97 - 1695645.17) - Avg length (360 - 109818.50)
# Trained
# Avg R (-9840.90 - 4121.21) - Avg length (1000 - 0.00)

# Stop reward
# Random
# Avg R (-909.51 - 1007706.35) - Avg length (258 - 63487.68)
# Trained
# Avg R (1102.71 - 1450089.28) - Avg length (218 - 36554.24)