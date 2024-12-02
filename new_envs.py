import random
from torch import tensor as tt
from torch.distributions import Categorical
from torch.nn.functional import one_hot as one_hot
import pickle
import torch
import utils
import numpy as np
import specs
import os
from environments import mg1
from environments import mg2
from environments import mg3

# use GPU if available
if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")

# Environment wrapper
class Environment:
    def __init__(self, name, kind, model):
        if not(kind != 'mg' or kind!= 'mmg' or kind != 'overcooked'):
            print("Error: Environment kind must be \'mg\' or \'mmg\' or \'overcooked\'")
            return
        else:
            self.name = name
            self.kind = kind
            self.model = model
            self.state = model.state

    def reset(self):
        self.state = self.model.initial(self.model.state_space)
        return self.state
    
    def step(self, joint_action):
        if self.kind == 'mg':
            model_joint_action = tuple([action_space[a.int()] for action_space, a in zip(self.model.action_spaces, joint_action)])

        elif self.kind == 'mmg':
            model_joint_action = tuple([int(a) for a in joint_action])

        elif self.kind == 'overcooked':
            model_joint_action = [self.model.action_space[joint_action[0]], self.model.action_space[joint_action[1]]]
 
        self.state = self.model.transition(self.state, model_joint_action)
        done = False
        return self.state, done
    
    def featurise(self, state):

        if self.kind == 'mg':
            features = self.model.featurise(state)
        elif self.kind == 'mmg':
            features = state
        elif self.kind == 'overcooked':
            features = self.model.featurise(state)

        return features
    
    def label(self, state):

        if self.kind == 'mg':
            labels = self.model.label(state)
        elif self.kind == 'mmg':
            labels = self.model.label(state)
        elif self.kind == 'overcooked':
            label_dict = self.model.label(state)
            labels = []
            for k in label_dict.keys():
                if label_dict[k][0]:
                    labels.append('0_' + k)
                if label_dict[k][1]:
                    labels.append('1_' + k)

        return labels
    
    def get_obs_size(self):

        if self.kind == 'mg':
            obs_size = self.model.num_states
        elif self.kind == 'mmg':
            obs_size = self.model.state_size
        elif self.kind == 'overcooked':
            # obs_size = len(featurise(self.state))
            obs_size = 62

        return obs_size

    def get_act_sizes(self):

        if self.kind == 'mg':
            act_sizes = [len(a) for a in self.model.action_spaces]
        elif self.kind == 'mmg':
            act_sizes = self.model.action_sizes
        elif self.kind == 'overcooked':
            act_sizes = [6,6]

        return act_sizes

    def get_name(self):

        return self.name

    def get_kind(self):

        return self.kind

    def save(self):

        with open("models/{}/{}.pickle".format(self.kind, self.name), 'wb') as f:
            pickle.dump(self, f)


class MarkovGame:
    def __init__(self, num_players, state_space, action_spaces, initial, transition, labeller):
        self.num_players = num_players
        self.state_space = state_space
        self.num_states = len(state_space)
        self.action_spaces = action_spaces
        self.initial = initial
        self.transition = transition
        self.labeller = labeller
        self.state = initial(state_space)

    def step(self, joint_action):
        if len(joint_action) != self.num_players:
            print("Error: Joint action must have length equal to number of players")
            return
        else:
            for i in range(self.num_players):
                assert joint_action[i] in self.action_spaces[i] 
            self.state = self.transition(self.state, joint_action)
            return self.state
        
    def reset(self):
        self.state = self.initial(self.state_space)
        return self.state
    
    def featurise(self, state):
        return one_hot(state, self.num_states)
    
    def label(self, state):
        return self.labeller(state)
    
    def print(self):
        print("State:", self.state)

    
mg_1 = MarkovGame(mg1.num_players, mg1.state_space, mg1.action_spaces, mg1.initial, mg1.transition, mg1.labeller)

mg_2 = MarkovGame(mg2.num_players, mg2.state_space, mg2.action_spaces, mg2.initial, mg2.transition, mg2.labeller)

mg_3 = MarkovGame(mg3.num_players, mg3.state_space, mg3.action_spaces, mg3.initial, mg3.transition, mg3.labeller)

