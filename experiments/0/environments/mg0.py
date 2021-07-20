import torch.tensor as tt
from torch.distributions import Categorical
import torch

if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")

### MARKOV GAME 0 ###
# Note that this game (and its variants) is for debugging purposes and so is very simple

### MARKOV GAME 0.0 ###
# Single player, single spec

# Number of players
num_players = 1

# State space
state_space = [tt(i) for i in range(2)]

# Action spaces
action_spaces = [('a','b')]

# Dists
d0 = Categorical(tt([ 0.5, 0.5 ]))
d1 = Categorical(tt([ 0.7, 0.3 ]))

# Initial state dist
def initial(states):
    return d1.sample()

# Transition function
def transition(s, a):
    if s == tt(0):
        if a == ('a',):
            return s
        else:
            return d0.sample()
    else:
        if a == ('a',):
            return s
        else:
            return d0.sample()

# Labelling function
def labeller(state):
    if state == tt(0):
        return ('phi',)
    else:
        return ('psi',)

### MARKOV GAME 0.1 ###
# Two players, single spec

# # Number of players
# num_players = 2

# # State space
# state_space = [tt(i) for i in range(2)]

# # Action spaces
# action_spaces = [('a','b'),('c','d')]

# # Dists
# d0 = Categorical(tt([ 0.5, 0.5 ]))
# d1 = Categorical(tt([ 0.7, 0.3 ]))

# # Initial state dist
# def initial(states):
#     return d1.sample()

# # Transition function
# def transition(s, a):
#     if s == tt(0):
#         if a == ('a','c'):
#             return s
#         else:
#             return d0.sample()
#     else:
#         if a == ('a','d'):
#             return s
#         else:
#             return d0.sample()

# # Labelling function
# def labeller(state):
#     if state == tt(0):
#         return ('phi',)
#     else:
#         return ('psi',)

### MARKOV GAME 0.2 ###
# Single player, two specs

# # Number of players
# num_players = 1

# # State space
# state_space = [tt(i) for i in range(2)]

# # Action spaces
# action_spaces = [('a','b')]

# # Dists
# d0 = Categorical(tt([ 0.5, 0.5 ]))
# d1 = Categorical(tt([ 0.7, 0.3 ]))

# # Initial state dist
# def initial(states):
#     return d1.sample()

# # Transition function
# def transition(s, a):
#     if s == tt(0):
#         if a == ('a',):
#             return s
#         else:
#             return d0.sample()
#     else:
#         if a == ('a',):
#             return s
#         else:
#             return d0.sample()

# # Labelling function
# def labeller(state):
#     if state == tt(0):
#         return ('phi',)
#     else:
#         return ('psi',)