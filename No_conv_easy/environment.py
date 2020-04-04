import numpy as np
import os
from PIL import Image



class Grid:
    def __init__(self):

        self.grid_shape = (7,7)

        # u:0, d:1, l:2, r:3
        self.action_space = ['u','d','l','r']
        self.n_actions = len(self.action_space)

        # state = [col (x),row (y),orie_id]
        # state_img = [height,width,depth] (shape)
        self.init_state = [0,0]
        self.goal_state = [1,3]
        
        self.state = self.init_state


    def get_renders(self):
        return self.statelbl_to_img, self.id_to_orie


    def reset(self):
        self.state = [0,0]
        return list(self.state)


    def step(self, action):

        # d(t-1) from the target
        d_curr = np.power(np.power(self.state[0]-self.goal_state[0],2)+np.power(self.state[1]-self.goal_state[1],2),0.5)

        # state
        if action == 0:   # up
            if self.state[1] != self.grid_shape[0]-1: self.state[1]+=1
        elif action == 1:   # down
            if self.state[1] != 0: self.state[1]-=1
        elif action == 2:   # left
            if self.state[0] != 0: self.state[0]-=1
        elif action == 3:   # right
            if self.state[0] != self.grid_shape[0]-1: self.state[0]+=1

        # d(t) from the target
        d_next = np.power(np.power(self.state[0]-self.goal_state[0],2)+np.power(self.state[1]-self.goal_state[1],2),0.5)

        if self.state == self.goal_state:
            reward = 50.0
            done = 1.0
        else:
            reward = 5*(d_curr-d_next)
            if reward == 0.0: reward = -1.0
            done = 0.0

        return list(self.state), reward, done


    def render(self):
        pass


