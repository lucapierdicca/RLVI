import numpy as np
import os
from PIL import Image



class Grid:
    def __init__(self):

        self.grid_shape = (7,7)

        # f:0, tl:1, tr:2
        self.action_space = ['f', 'tl', 'tr']
        self.n_actions = len(self.action_space)
        
        self.orie_to_id = {'E':0,'N':1,'O':2,'S':3}
        self.id_to_orie = {v:k for k,v in self.orie_to_id.items()}

        renders_path = './env_renders/'
        self.statelbl_to_img = { k[:3]:np.array(Image.open(renders_path+k)) \
            for k in os.listdir(renders_path)}
        print(self.statelbl_to_img.keys())

        # state = [row,col,orie_id]
        # state_img = [height,width,depth] (shape)
        self.init_state = [0,0,0]
        self.goal_state = [1,4,1]
        
        self.state = self.init_state

        self.turns = 0

    def get_renders(self):
        return self.statelbl_to_img, self.id_to_orie


    def reset(self):
        self.state = [0,0,0]
        self.turns = 0
        return list(self.state)

    def step(self, action):

        # state
        if action == 0:   # forward
            # E
            if self.state[2] == 0 and self.state[1] != self.grid_shape[1]-1: self.state[1]+=1
            # N
            elif self.state[2] == 1 and self.state[0] != self.grid_shape[0]-1: self.state[0]+=1
            # O
            elif self.state[2] == 2 and self.state[1] != 0: self.state[1]-=1
            # S
            elif self.state[2] == 3 and self.state[0] != 0: self.state[0]-=1
        else:
            if action == 1:   # turn left
                self.turns+=1
            elif action == 2:   # turn right
                self.turns-=1

            if self.turns >= 0:
                self.state[2] = abs(self.turns)%4
            else:
                self.state[2] = (4-(abs(self.turns)%4))%4

        
        # reward 
        if self.state == self.goal_state:
            reward = 10
            done = 1
        else:
            reward = -1
            done = 0

        return list(self.state), reward, done

    def render(self):
        pass


