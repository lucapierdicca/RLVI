import numpy as np
import os
from PIL import Image



class Grid:
    def __init__(self):

        self.grid_shape = (7,7)

        # u:0, tl:1, tr:2
        self.action_space = ['f','tl','tr']
        self.n_actions = len(self.action_space)

        # state = [col (x),row (y),orie_id]
        self.init_state = [0,0,0]
        self.goal_state = [1,3,1]
        
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

        # d(t) from the target
        d_next = np.power(np.power(self.state[0]-self.goal_state[0],2)+np.power(self.state[1]-self.goal_state[1],2),0.5)

        if self.state == self.goal_state:
            reward = 50.0
            done = 1.0
        else:
            reward = 5*(d_curr-d_next)
            done = 0.0

        return list(self.state), reward, done


    def render(self):
        pass

