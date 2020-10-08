from environment import Grid
import numpy as np
from pprint import pprint
import statistics as stat
import pickle

# init environment
env = Grid()

# init Q(s,a) table
q_table = {}

for i in range(env.grid_shape[0]):
    for j in range(env.grid_shape[1]):
        for k in range(4):
            q_table[(i,j,k)] = np.zeros(env.n_actions)


# Q-learning main loop---------------------
n_episode       = 5000
epsilon_start   = 1
epsilon_end     = 0.2
epsilon         = epsilon_start
alpha           = 0.6
gamma           = 0.4

for i in range(n_episode):
    s = env.reset()
    d = 0
    while d != 1:
        # epsilon greedy
        if np.random.uniform() < epsilon:
            # choose a random action
            a = np.random.randint(0, env.n_actions)
        else:
            # choose the greedy action
            a = np.argmax(q_table[tuple(s)])

        s_, r, d = env.step(a)
        q_table[tuple(s)][a] += alpha * (r + (gamma*max(q_table[tuple(s_)])*(1-d)) - q_table[tuple(s)][a])
        s = list(s_)

    epsilon =  -((epsilon_start - epsilon_end)/n_episode) * i + epsilon_start 
# -----------------------------------------


# let's plot the policy over the grid
import matplotlib.pyplot as plt

for s,v in q_table.items():
    if s[0] != 4 or s[1] != 5:
        x = s[0]+0.4  
        y = s[1]+0.4
        if s[2] == 0: x+=0.2
        elif s[2] == 1: y+=0.2
        elif s[2] == 2: x-=0.2
        elif s[2] == 3: y-=0.2
        max_a = np.argmax(v) 
        
        plt.text(x, y, env.action_space[max_a])

plt.scatter(4+0.5, 5+0.5, color="cyan", marker='s', s=150)

plt.xlim([0,7])
plt.ylim([0,7])
plt.grid(True)
plt.show()


# ---------------------------------------------


#evaluation
n_episode = 100
epsilon = 0.01
max_episode = 50

data = {'trained':[],'random':[]}

for p in data.keys():
    b_r = []
    b_sc = []
    trajectories = []

    for episode in range(n_episode):
        episode_reward=0
        episode_step_counter=0
        t = []

        s = env.reset()
        t.append(s)
        d = 0

        while True:
            if p == 'random': 
                a = np.random.randint(0, env.n_actions)
            else:
                if np.random.uniform() < epsilon:
                    a = np.random.randint(0, env.n_actions)
                else:                   
                    a = np.argmax(q_table[tuple(s)])

            s_, r, d = env.step(a)
            s = list(s_)

            t.append(s)
            episode_reward+=r
            episode_step_counter+=1
        
            if d or episode_step_counter == max_episode-1:
                break

        
        print("policy: %s - episode: %d - reward: %d - step: %d" % (p,episode+1,episode_reward,episode_step_counter+1))
        b_r.append(episode_reward)
        b_sc.append(episode_step_counter+1)
        trajectories.append(t)


    data[p] = [b_r,b_sc,trajectories]


print("Random")
print("Avg R (%.2f - %.2f) - Avg length (%d - %.2f)" % (stat.mean(data['random'][0]),stat.variance(data['random'][0]),
                                                        stat.mean(data['random'][1]),stat.variance(data['random'][1])))

print("Trained")
print("Avg R (%.2f - %.2f) - Avg length (%d - %.2f)" % (stat.mean(data['trained'][0]),stat.variance(data['trained'][0]),
                                                        stat.mean(data['trained'][1]),stat.variance(data['trained'][1])))



    
