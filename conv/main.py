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
        env.reset_reward_h()

        _, max_Q, argmax_Q = agent.get_action(h, statelbl_to_img, id_to_orie)

        while True:
            
            a, _,_ = agent.get_action(h, statelbl_to_img, id_to_orie)

            s_, r, d = env.step(a)
            #print(r)
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
    data = {'trained':[],'random':[]}
    
    for p in data.keys():
        b_r = []
        b_sc = []
        trajectories = []

        for episode in range(n_episode):
            episode_reward=0
            episode_step_counter=0
            t = []
            
            reset_history()
            s = env.reset()
            history.append(s)
            t.append(s)

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

                t.append(s_)
                
                episode_reward+=r
                episode_step_counter+=1

                if d or episode_step_counter == max_episode-1:
                    break
            
            print("policy: %s - episode: %d - reward: %d - step: %d" % (p,episode+1,episode_reward,episode_step_counter+1))
            b_r.append(episode_reward)
            b_sc.append(episode_step_counter+1)
            trajectories.append(t)


        data[p] = [b_r,b_sc,trajectories]

    return data

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
                n_history,
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
    max_episode = 50
    epsilon = 0.05

    # start evaluation
    session,Q_op,h_ph = DQN.restore()

    data = eval_loop(session,Q_op,h_ph,
                             n_episode,
                             max_episode,
                             epsilon)

    print("Random")
    print("Avg R (%.2f - %.2f) - Avg length (%d - %.2f)" % (stat.mean(data['random'][0]),stat.variance(data['random'][0]),
                                                            stat.mean(data['random'][1]),stat.variance(data['random'][1])))

    print("Trained")
    print("Avg R (%.2f - %.2f) - Avg length (%d - %.2f)" % (stat.mean(data['trained'][0]),stat.variance(data['trained'][0]),
                                                            stat.mean(data['trained'][1]),stat.variance(data['trained'][1])))



    #graphs
    import matplotlib.pyplot as plt

    trajectories = data['trained'][2]
    max_len = max(data['trained'][1])

    for i in range(len(trajectories)):
        if len(trajectories[i])<max_len:
            pad = [[-1,-1,-1]]*(max_len-len(trajectories[i]))
            trajectories[i] = trajectories[i]+pad
        
    trajectories = np.array(trajectories)



    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_aspect(aspect='equal')
    

    for t in trajectories:
        x = t[:,0]; x = x[x != -1]; x = ((x+0.5)-0.2)+0.4*np.random.uniform();
        y = t[:,1]; y = y[y != -1]; y = ((y+0.5)-0.2)+0.4*np.random.uniform();
        ax.plot( x, 
                 y,
                 '-',
                 c='blue',
                 linewidth=0.5,
                 alpha =0.1)

    # modal_t = []
    # for t in range(max_len):
    #     xy = trajectories[:,t,:2]; xy = xy[xy[:,0]!=-1];
    #     xy_count = {}
    #     for i in xy:
    #         if tuple(i) in xy_count: xy_count[tuple(i)] +=1
    #         else: xy_count[tuple(i)]=1
    #     xy = sorted(xy_count.items(), key=lambda items: items[1])
    #     modal_t.append(xy[-1][0])


    plt.xlim([0,7])
    plt.ylim([0,7])
    plt.grid(True)
    plt.title('Position trajectories')
    

    fig2 = plt.figure()
    
    ax = fig2.add_subplot(311)
    #ax.set_xlabel('step')
    ax.set_ylim([0,6])
    ax.set_xlim([0,50])
    ax.set_ylabel('x')
    for t in trajectories:
        x = t[:,0]; x = x[x != -1]
        ax.plot(x,'-',c='blue',alpha = 0.1,linewidth=0.5)

    ax = fig2.add_subplot(312)
    ax.set_ylim([0,6])
    ax.set_xlim([0,50])
    ax.set_ylabel('y')
    for t in trajectories:
        y = t[:,1]; y = y[y != -1]
        ax.plot(y,'-',c='blue',alpha = 0.1,linewidth=0.5)
    
    ax = fig2.add_subplot(313)
    ax.set_ylim([0,3])
    ax.set_xlim([0,50])
    ax.set_ylabel('theta')
    for t in trajectories:
        theta = t[:,2]; theta = theta[theta != -1];
        ax.plot(theta,'-',c='blue',alpha = 0.1,linewidth=0.5)   

    
   
    plt.show()