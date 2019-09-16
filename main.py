from environment import Grid
from network import DQN
from collections import deque



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
    hist_episode_reward=[]

    for episode in range(n_episode):

        reset_history()
        s = env.reset()
        history.append(s)
        h = list(history)

        episode_step_counter, episode_reward = 0,0
        while True:
            # env.render()
            a = agent.choose_action(h, 
                statelbl_to_img, 
                id_to_orie)
            
            s_, r, d = env.step(a)
            history.append(s_)
            h_ = list(history)

            episode_reward+=r
            # a transition is [[history],int,int,[history_],int]
            agent.store_transition(h, a, r, h_, d)
            print("%d - %d - %d - %s - %s - %d - %s - %d - %d" % 
                (episode, 
                 tot_step_counter, 
                 episode_step_counter, 
                 str(h), 
                 id_to_action[a], 
                 r, 
                 str(h_), 
                 d, 
                 episode_reward))

            if (tot_step_counter > 200) and (tot_step_counter % 5 == 0):
                
                cost = agent.train(statelbl_to_img, id_to_orie)
                print("------> %f" % cost)
            
            h = list(h_)

            tot_step_counter += 1
            episode_step_counter+=1
            
            if d or episode_step_counter == 500:
                hist_episode_reward.append(episode_reward) 
                break
            
            

    print('Game over')
    print(hist_episode_reward)
    #agent.plot_cost()


#--------------------------------------------

env = Grid()

n_history = 2
history = deque([], maxlen=n_history)

agent = DQN(env.n_actions,
            n_history,
            learning_rate=0.01,
            gamma=0.9,
            epsilon=1.0,
            replace_target_iter=200,
            memory_size=10000,
            batch_size=32,
            hidden_units=256)
    

n_episode = 1000  
train_loop(n_episode)