from environment import Grid
from network import DQN
from collections import deque
import pickle
from time import gmtime, strftime



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
            # env.render()
            a, _, _ = agent.get_action(s)

            s_, r, d = env.step(a)

            # a transition is [[history],int,int,[history_],int]
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
            agent.saver.save(agent.sess, "./weights/weights.ckpt",
                 global_step=episode, write_meta_graph=False)
            print('********** SAVE *********')

        #epsilon annealing
        if tot_step_counter > 20000:
            agent.epsilon = agent.epsilon - 0.9/110000


            
    pickle.dump(histories, open('histories.pickle','wb'))
    print('Game over')




#--------------------------------------------

env = Grid()


agent = DQN(env.n_actions,
            learning_rate=0.000001, #0.1
            gamma=0.99,
            epsilon=1.0,
            memory_size=1000000,
            batch_size=64,
            hidden_units=128)
    

n_episode = 50000
offset_train = 1
offset_copy = 300
max_episode = 1000

train_loop(n_episode, 
    offset_train, 
    offset_copy,
    max_episode)

#DQN.eval()