from torch import tensor as tt
from torch.distributions import Categorical
import torch

if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")

### MARKOV GAME 2 ###

# Number of players
num_players = 2

# State space
state_space = [tt(i) for i in range(3)]

# Action spaces
action_spaces = [('a','b'),('x','y')]

# Dists
d0 = Categorical(tt([ 1.0, 0.0, 0.0 ]))
d1 = Categorical(tt([ 0.0, 1.0, 0.0 ]))
d2 = Categorical(tt([ 0.0, 0.0, 1.0 ]))
d3 = Categorical(tt([ 0.0, 0.1, 0.9 ]))
d4 = Categorical(tt([ 0.0, 0.4, 0.6 ]))

# Initial state dist
def initial(states):
    return d1.sample()

# Transition function
def transition(s, a):
    if s == tt(0):
        if a[0] == ('a'):
            return d0.sample()
        else:
            return d3.sample()
    elif s == tt(1):
        return d4.sample()
    else:
        if a == ('a','x'):
            return d1.sample()
        elif a == ('b','x'):
            return d2.sample()
        else:
            return d0.sample()

# Label function
def labeller(s):
    if s == tt(0):
        return ('psi')
    elif s == tt(1):
        return ('chi','phi')
    else:
        return ()
