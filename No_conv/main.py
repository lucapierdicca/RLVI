from environment import Grid
from network import DQN
from collections import deque
import pickle
from time import gmtime, strftime



def train_loop(n_episode, offset_train, offset_copy, max_episode):

    global env

    tot_step_counter=0
    cost=0.0
    #epsilon_counter = 0
    action_space = {'u':0, 'd':1, 'l':2, 'r':3}
    id_to_action = {v:k for k,v in action_space.items()}

    histories = {'episode_reward':[],
                 'max_Q':[],
                 'cost':[],
                 'epsilon':[],
                 'episode_len':[]}

    for episode in range(n_episode):

        s = env.reset()

        episode_step_counter=0.0 
        episode_reward=0.0
        episode_max_Q=0.0
        episode_cost=0.0
        #episode_epsilon=0.0

        while True:

            # env.render()
            a, max_Q = agent.get_action(s)

            s_, r, d = env.step(a)

            # a transition is [[history],int,int,[history_],int]
            agent.store_transition(s, a, r, s_, d)
            #print(s,id_to_action[a],r,s_,d)
            # print("%d - %d - %d - %s - %s - %f - %s - %f - %f" % 
            #     (episode, 
            #      tot_step_counter, 
            #      episode_step_counter, 
            #      str(h), 
            #      id_to_action[a], 
            #      max_Q, 
            #      str(h_), 
            #      d, 
            #      r))

            if (tot_step_counter > 10000) and (tot_step_counter % offset_train == 0):
                cost = agent.train(episode)
                #print('********* TRAIN ********')

            if (tot_step_counter > 10000) and (tot_step_counter % offset_copy == 0):
                agent.copy_vars()
                #print('********* COPY *********')

            s = list(s_)

            #episode_epsilon+=agent.epsilon
            episode_max_Q+=max_Q
            episode_reward+=r
            episode_cost+=cost

            episode_step_counter+=1
            tot_step_counter += 1
            
   
            if d or episode_step_counter == max_episode:
                print("episode: %d - goal: %d (%d) - e: %.4f (%d) - Q: %.5f - loss: %.3f - reward: %.1f" % 
                    (episode+1, d, episode_step_counter,
                        agent.epsilon, tot_step_counter, 
                        episode_max_Q/episode_step_counter, 
                        episode_cost/episode_step_counter, 
                        episode_reward/episode_step_counter))
                histories['max_Q'].append(episode_max_Q/episode_step_counter)
                histories['cost'].append(episode_cost/episode_step_counter)
                histories['episode_reward'].append(episode_reward/episode_step_counter)
                histories['epsilon'].append(agent.epsilon)
                histories['episode_len'].append(episode_step_counter)
                break
        
        if tot_step_counter > 20000:
            #delta = n_episode-100
            agent.epsilon = agent.epsilon - 0.9/200000 #max(((-0.9/delta)*epsilon_counter) + 1.0, 0.3)   
            #epsilon_counter+=1
            
    pickle.dump(histories, open('histories.pickle','wb'))
    print('Game over')
    #agent.plot_cost()




#--------------------------------------------

# env = Grid()


# agent = DQN(env.n_actions,
#             learning_rate=0.000001, #0.1
#             gamma=0.99,
#             epsilon=1.0,
#             memory_size=1000000,
#             batch_size=64,
#             hidden_units=128)
    

# n_episode = 50000
# offset_train = 30
# offset_copy = 300
# max_episode = 1000

# train_loop(n_episode, 
#     offset_train, 
#     offset_copy,
#     max_episode)

DQN.eval()