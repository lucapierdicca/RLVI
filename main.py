from environment import Grid
from network import DQN
from collections import deque
import pickle
from time import gmtime, strftime


def reset_history():

    global history
    for i in range(history.maxlen):
        history.append(['_','_',0])


def train_loop(n_episode, offset_train, offset_copy, max_episode):

    global env
    statelbl_to_img, id_to_orie = env.get_renders()
    action_space = {'f':0, 'tl':1, 'tr':2}
    id_to_action = {v:k for k,v in action_space.items()}

    tot_step_counter=0
    cost=0.0

    histories = {'episode_reward':[],
                 'max_Q':[],
                 'cost':[],
                 'epsilon':[]}

    for episode in range(n_episode):

        reset_history()
        s = env.reset()
        history.append(s)
        h = list(history)

        episode_step_counter=0.0 
        episode_reward=0.0
        episode_max_Q=0.0
        episode_cost=0.0
        episode_epsilon=0.0

        while True:

            # env.render()
            a, max_Q = agent.get_action(h, statelbl_to_img, id_to_orie)

            s_, r, d = env.step(a)
            history.append(s_)
            h_ = list(history)

            # a transition is [[history],int,int,[history_],int]
            agent.store_transition(h, a, r, h_, d)
            # print("%d - %d - %d - %s - %s - %d - %s - %d - %d - %f" % 
            #     (episode, 
            #      tot_step_counter, 
            #      episode_step_counter, 
            #      str(h), 
            #      id_to_action[a], 
            #      r, 
            #      str(h_), 
            #      d, 
            #      episode_reward,
            #      max_Q))

            if (tot_step_counter > 10000) and (tot_step_counter % offset_train == 0):
                cost = agent.train(statelbl_to_img, id_to_orie)
                #print('********* TRAIN ********')

            if (tot_step_counter > 10000) and (tot_step_counter % offset_copy == 0):
                agent.copy_vars()
                #print('********* COPY *********')

            h = list(h_)

            episode_epsilon+=agent.epsilon
            episode_max_Q+=max_Q
            episode_reward+=r
            episode_cost+=cost

            episode_step_counter+=1
            tot_step_counter += 1
            
   
            if d or episode_step_counter == max_episode:
                print("episode: %d - goal: %d (%d) - e: %.4f (%d) - Q: %.2f - loss: %.3f - reward: %.1f" % 
                    (episode+1, d, episode_step_counter,
                        episode_epsilon/episode_step_counter, tot_step_counter, 
                        episode_max_Q/episode_step_counter, 
                        episode_cost/episode_step_counter, 
                        episode_reward))
                histories['max_Q'].append(episode_max_Q/episode_step_counter)
                histories['cost'].append(episode_cost/episode_step_counter)
                histories['episode_reward'].append(episode_reward)
                histories['epsilon'].append(episode_epsilon/episode_step_counter)
                break
            
            
    pickle.dump(histories, open('histories.pickle','wb'))
    print('Game over')
    #agent.plot_cost()


#--------------------------------------------

env = Grid()

n_history = 4
history = deque([], maxlen=n_history)

agent = DQN(env.n_actions,
            n_history,
            learning_rate=0.00025, #0.1
            gamma=0.99,
            epsilon=1.0,
            memory_size=500000,
            batch_size=32,
            hidden_units=256)
    

n_episode = 2000
offset_train = 5
offset_copy = 200
max_episode = 1000

train_loop(n_episode, 
    offset_train, 
    offset_copy,
    max_episode)