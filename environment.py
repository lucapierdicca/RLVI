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
        self.statelbl_to_img = {}
        for filename in os.listdir(renders_path):
            img = Image.open(renders_path+filename)
            img = img.convert('L')
            img = img.resize((84,84), Image.ANTIALIAS)
            img = np.array(img).reshape((84,84,1))/255
            self.statelbl_to_img[filename[:3]] = img
        print(self.statelbl_to_img.keys())

        # state = [row,col,orie_id]
        # state_img = [height,width,depth] (shape)
        self.init_state = [0,0,0]
        self.goal_state = [1,3,1]
        
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
        distance = np.power(np.power(self.state[0]-self.goal_state[0],2)+np.power(self.state[1]-self.goal_state[1],2),0.5)
        reward = np.exp(-10*distance)

        if self.state == self.goal_state:
            #reward = 10.0
            done = 1.0
        else:
            #reward = -0.01
            done = 0.0

         

        return list(self.state), reward, done

    def render(self):
        pass


