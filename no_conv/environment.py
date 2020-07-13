import numpy as np
import os
from PIL import Image



class Grid:
    def __init__(self):

        self.grid_shape = (7,7)

        # u:0, tl:1, tr:2
        self.action_space = ['f','tl','tr']
        self.n_actions = len(self.action_space)
        
        self.orie_to_id = {'E':0,'N':1,'O':2,'S':3}
        self.id_to_orie = {v:k for k,v in self.orie_to_id.items()}

        renders_path = './env_renders/'
        self.statelbl_to_img = {}

        for filename in os.listdir(renders_path):
            cyan = 0
            if len(filename) > 7: cyan = 1
            self.statelbl_to_img[filename[:3]] = cyan
        # state = [col (x),row (y),orie_id]
        self.init_state = [0,0,0]
        self.goal_state = [1,3,1]

        lr = np.array([6,0])
        g = np.array([self.goal_state[0],self.goal_state[1]])
        self.d = np.linalg.norm(g-lr)

        self.o = {  0:np.array([[1],[0]]),
                    1:np.array([[0],[1]]),
                    2:np.array([[-1],[0]]),
                    3:np.array([[0],[-1]])}
        
        self.state = self.init_state


    def get_renders(self):
        return self.statelbl_to_img, self.id_to_orie


    def reset(self):
        self.state = [0,0,0]
        return list(self.state)


    def step(self, action):

        # d(t-1) from the target
        d_curr = np.power(np.power(self.state[0]-self.goal_state[0],2)+np.power(self.state[1]-self.goal_state[1],2),0.5)

        # state
        if action == 0:   # forward
            # E
            if self.state[2] == 0 and self.state[0] != self.grid_shape[0]-1: self.state[0]+=1
            # N
            elif self.state[2] == 1 and self.state[1] != self.grid_shape[1]-1: self.state[1]+=1
            # O
            elif self.state[2] == 2 and self.state[0] != 0: self.state[0]-=1
            # S
            elif self.state[2] == 3 and self.state[1] != 0: self.state[1]-=1
        else:
            if action == 1:   # turn left
                if self.state[2] == 0: self.state[2] = 1
                elif self.state[2] == 1: self.state[2] = 2
                elif self.state[2] == 2: self.state[2] = 3
                elif self.state[2] == 3: self.state[2] = 0
            elif action == 2:   # turn right
                if self.state[2] == 0: self.state[2] = 3
                elif self.state[2] == 1: self.state[2] = 0
                elif self.state[2] == 2: self.state[2] = 1
                elif self.state[2] == 3: self.state[2] = 2

        # # d(t) from the target
        # d_next = np.power(np.power(self.state[0]-self.goal_state[0],2)+np.power(self.state[1]-self.goal_state[1],2),0.5)

        # if self.state == self.goal_state:
        #     reward = 50.0
        #     done = 1.0
        # else:
        #     reward = 5*(d_curr-d_next)
        #     #--------
        #     if reward == 0:
        #         cyan = self.statelbl_to_img[str(self.state[0])+str(self.state[1])+self.id_to_orie[self.state[2]]]
        #         if cyan: reward = +10
        #         else: reward = -10
        #     #-------
        #     done = 0.0

        x = self.state[0]-1
        y = self.state[1]-3
        m = -20.0
        alpha=0.2
        reward = m/np.power(self.d,2)*(np.power(x,2)+np.power(y,2))
        reward_orie = 0

        g = np.array([[(2*m*x)/np.power(self.d,2)],[(2*m*y)/np.power(self.d,2)]])
        
        if x==0 and y==0:
            reward = 1
            g=np.array([[0],[1]])
            alpha=-4

        g = g/np.linalg.norm(g)
        reward_orie = alpha*reward*(-0.5*np.dot(self.o[self.state[2]].T,g)[0][0]+0.5)

        reward+=reward_orie

        if self.state == self.goal_state:
            done = 1.0
        else:
            done = 0.0
        
        return list(self.state), reward, done


    def render(self):
        pass


# def r(l):
#     return range(l[0],l[1]+1) 

# a = [0,10]
# b = [1,9]
# c = [2,8]
# d = [3,7]
# e = [4,6]
# f = 5
# states = {0:[],1:[],2:[],3:[],4:[],5:[]}
# values = {0:[],1:[],2:[],3:[],4:[],5:[]}

# for i in range(7):
#     for j in range(7):
#         for k in range(4):
#             if (i+4 in a and j+2 in r(a)) or ((j+2 in a and i+4 in r(a))):
#                 states[0].append([i,j,k])
#             elif (i+4 in b and j+2 in r(b)) or ((j+2 in b and i+4 in r(b))):
#                 states[1].append([i,j,k])
#             elif (i+4 in c and j+2 in r(c)) or ((j+2 in c and i+4 in r(c))):
#                 states[2].append([i,j,k])
#             elif (i+4 in d and j+2 in r(d)) or ((j+2 in d and i+4 in r(d))):
#                 states[3].append([i,j,k])
#             elif (i+4 in e and j+2 in r(e)) or ((j+2 in e and i+4 in r(e))):
#                 states[4].append([i,j,k])
#             elif i+4 == 5 and j+2 == 5:
#                 states[5].append([i,j,k])


# from pprint import pprint
# pprint(states)
# m = -20
# goal_state = [1,3,1]

# lr = np.array([6,0])
# g = np.array([goal_state[0],goal_state[1]])
# d = np.linalg.norm(g-lr)
# print(np.power(d,2))

# o = {  0:np.array([[1],[0]]),
#             1:np.array([[0],[1]]),
#             2:np.array([[-1],[0]]),
#             3:np.array([[0],[-1]])}

# #f = m/np.power(d,2)*(np.power(x,2)+np.power(y,2))


# for k,v in states.items():
#     for s in v:
#         x = s[0]-1
#         y = s[1]-3
        
#         df = np.array([[(2*m*x)/np.power(d,2)],[(2*m*y)/np.power(d,2)]])
        
#         if x==0 and y==0:
#             df=np.array([[0],[1]])

#         reward = np.dot(o[s[2]].T,df)[0][0]

#         values[k].append(reward)

# import matplotlib.pyplot as plt

# for k,v in values.items():
#     plt.plot(v)

# plt.show()
