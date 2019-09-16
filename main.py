from environment import Grid
from network import DQN
from collections import deque
import pickle
from time import gmtime, strftime


def reset_history():

    global history
    for i in range(history.maxlen):
        history.append(['_','_',0])

def train_loop(n_episode):

    global env
    statelbl_to_img, id_to_orie = env.get_renders()
    action_space = {'f':0, 'tl':1, 'tr':2}
    id_to_action = {v:k for k,v in action_space.items()}

    tot_step_counter=0
    cost = 0.0

    histories = {'episode_reward':[],
                 'max_Q':[],
                 'cost':[],
                 'epsilon':[]}

    for episode in range(n_episode):

        reset_history()
        s = env.reset()
        history.append(s)
        h = list(history)

        episode_step_counter, episode_reward, episode_max_Q, episode_cost, episode_epsilon = 0.0,0.0,0.0,0.0,0.0
        while True:

            episode_epsilon+=agent.epsilon

            # env.render()
            a, max_Q = agent.choose_action(h, 
                statelbl_to_img, 
                id_to_orie)

            episode_max_Q+=max_Q
            
            s_, r, d = env.step(a)
            history.append(s_)
            h_ = list(history)

            episode_reward+=r

            # a transition is [[history],int,int,[history_],int]
            agent.store_transition(h, a, r, h_, d)
            print("%d - %d - %d - %s - %s - %d - %s - %d - %d - %f" % 
                (episode, 
                 tot_step_counter, 
                 episode_step_counter, 
                 str(h), 
                 id_to_action[a], 
                 r, 
                 str(h_), 
                 d, 
                 episode_reward,
                 max_Q))

            if (tot_step_counter > 5000) and (tot_step_counter % 5 == 0):
                cost = agent.train(statelbl_to_img, id_to_orie)
                print("------> %f" % cost)

            episode_cost+=cost
            
            h = list(h_)

            tot_step_counter += 1
            episode_step_counter+=1

   
            if d or episode_step_counter == 200:
                histories['max_Q'].append(episode_max_Q/episode_step_counter)
                histories['cost'].append(episode_cost/episode_step_counter)
                histories['episode_reward'].append(episode_reward/episode_step_counter)
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
            replace_target_iter=200,
            memory_size=500000,
            batch_size=32,
            hidden_units=256)
    

n_episode = 5000 
train_loop(n_episode)