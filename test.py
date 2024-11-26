from subprocess import check_output
from specs import *

out=check_output(['rabinizer4/bin/ltl2ldba', '-e', 'G ( a -> F b )'])
print(out)
l = LDBA('G ( a -> F b )')
print(l.get_num_states())
print(l.get_num_eps_actions())
print(l.acc)
print(l.eps)
print(l.delta)