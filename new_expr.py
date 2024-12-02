import learners
import specs
import new_envs
from os import path
import pickle
import copy
import random
import sys
import torch
import numpy as np
import copy
import os

def run(hp, filename = None, modelname= None):
    print("into run")
    env = new_envs.Environment(hp['env_name'], hp['env_type'], hp['env'])
    sps = hp['specs']
    specifications = [sp[0] for sp in sps]
    weights = [sp[1] for sp in sps]
    print(f"weights: {weights}")
    
    spec = []
    for s in list(zip(specifications, weights)):
        f = "experiments/0/specs/" + s[0] + '.pickle'
        if path.isfile(f):
            os.remove(f)
            new_spec = specs.Spec(s[0], s[1])
            new_spec.save("experiments/0/specs")
            spec.append(new_spec)
        else:
            new_spec = specs.Spec(s[0], s[1])
            new_spec.save("experiments/0/specs")
            spec.append(new_spec)
            # hp['specs'] = spec
    model_constants = { 'discounts': hp['discounts'],
                        'l2_reg': hp['nat_grad_l2_regulariser_weight'],
                        'lrs': hp['learning_rates'] }
    
    almanac = learners.Almanac(env, spec, hp['optimisers'], hp['buffers'], hp['models'], hp['local'], model_constants)
    train_constants = { 'continue_prob': hp['continue_prob'],
                        'epsilon': hp['epsilon'],
                        'nat_grad_tolerance': hp['nat_grad_convergence_tolerance'],
                        'neg_ent_reg': hp['actor_neg_entropy_regulariser_weight'],
                        'non_det_reg': hp['actor_nondeterminism_regulariser_weight'],
                        'sum_val_reg': hp['critic_sum_value_regulariser_weight'],
                        'a_neg_var_reg': hp['actor_neg_variance_regulariser_weight'],
                        'c_neg_var_reg': hp['critic_neg_variance_regulariser_weight'],
                        'max_nat_grad_norm': hp['max_nat_grad_norm'],
                        'max_critic_norm': hp['max_critic_norm'],
                        'reward_weight': hp['reward_weight'] }
    run_num = random.randint(0, 5)
    filename = 'experiments/0/scores/almanac-{}.txt'.format(run_num)
    directory = os.path.dirname(filename)

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(filename,'w') as f:
        f.write(" ")
    trained, _, score_at_end, _ = almanac.train(hp['steps'], env, hp['actual_state_dist'], hp['patient_updates'], train_constants, hp['run_id'], filename)
    if modelname != None:
        almanac.lrs = None
        almanac.patient_buffer = None
        almanac.hasty_buffer = None
        with open(modelname, 'wb') as f:
            pickle.dump(almanac, f)

    final_score = 0
    print("Score: {}".format(score_at_end))
    
    return trained, final_score, score_at_end

# MG 1
exp_1_1 = { 'actor_neg_entropy_regulariser_weight': 0.1,
            'actor_neg_variance_regulariser_weight': 5.0,
            'actor_nondeterminism_regulariser_weight': 0.005,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.9,
            'critic_sum_value_regulariser_weight': 0.5,
            'critic_neg_variance_regulariser_weight': 0.5,
            'discounts': { 'hasty': 0.8,
                           'patient': 0.9 },
            'env': new_envs.mg_1,
            'env_name': 'exp1_mg1',
            'env_type': 'mg',
            'epsilon': lambda e: 0.1,
            'nat_grad_l2_regulariser_weight': 0.001,
            'labeller': new_envs.mg_1.label,
            'learning_rates': { 'actor': ('constant', 0.05),
                                'critic': ('constant', 0.1),
                                'patient_nat_grad': ('constant', 0.1),
                                'hasty_nat_grad': ('constant', 0.1),
                                'mu': ('constant', 0.1) },
            'local': True,
            'nat_grad_convergence_tolerance': 0.05,
            'max_critic_norm': 10.0,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [16, 24, 16]),
                        'critic': ('dnn', [16, 24, 16]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'run_id': None,
            'patient_updates': True,
            'reward_weight': 10,
            'specs': [ ('G chi', 1),
                       ('F phi', 1),
                       ('G F psi', 1) ],
            'steps': 30000 }

exp_1_2 = { 'actor_neg_entropy_regulariser_weight': 0.1,
            'actor_neg_variance_regulariser_weight': 5.0,
            'actor_nondeterminism_regulariser_weight': 0.005,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.9,
            'critic_sum_value_regulariser_weight': 0.5,
            'critic_neg_variance_regulariser_weight': 0.5,
            'discounts': { 'hasty': 0.8,
                           'patient': 0.9 },
            'env': new_envs.mg_2,
            'env_name': 'exp1_mg2',
            'env_type': 'mg',
            'epsilon': lambda e: 0.1,
            'nat_grad_l2_regulariser_weight': 0.001,
            'labeller': new_envs.mg_2.label,
            'learning_rates': { 'actor': ('constant', 0.05),
                                'critic': ('constant', 0.1),
                                'patient_nat_grad': ('constant', 0.1),
                                'hasty_nat_grad': ('constant', 0.1),
                                'mu': ('constant', 0.1) },
            'local': True,
            'nat_grad_convergence_tolerance': 0.05,
            'max_critic_norm': 10.0,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [16, 24, 16]),
                        'critic': ('dnn', [16, 24, 16]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'run_id': None,
            'patient_updates': True,
            'reward_weight': 1,
            'specs': [ ('G (chi -> F psi)', 0.4),
                        ('G F phi', 0.6) ],
            'steps': 1000 }

exp_1_3 = { 'actor_neg_entropy_regulariser_weight': 0.1,
            'actor_neg_variance_regulariser_weight': 5.0,
            'actor_nondeterminism_regulariser_weight': 0.005,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.9,
            'critic_sum_value_regulariser_weight': 0.5,
            'critic_neg_variance_regulariser_weight': 0.5,
            'discounts': { 'hasty': 0.8,
                           'patient': 0.9 },
            'env': new_envs.mg_3,
            'env_name': 'exp1_mg3',
            'env_type': 'mg',
            'epsilon': lambda e: 0.1,
            'nat_grad_l2_regulariser_weight': 0.001,
            'labeller': new_envs.mg_3.label,
            'learning_rates': { 'actor': ('constant', 0.05),
                                'critic': ('constant', 0.1),
                                'patient_nat_grad': ('constant', 0.1),
                                'hasty_nat_grad': ('constant', 0.1),
                                'mu': ('constant', 0.1) },
            'local': True,
            'nat_grad_convergence_tolerance': 0.05,
            'max_critic_norm': 10.0,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [16, 24, 16]),
                        'critic': ('dnn', [16, 24, 16]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'run_id': None,
            'patient_updates': True,
            'reward_weight': 1,
            'specs': [ ('G (chi -> F psi)', 0.3),
                        ('G F phi', 0.7) ],
            'steps': 1000 }


def expr():
    # run(exp_1_1)
    completed,_,_ = run(exp_1_3)
    print("completed")

if __name__ == '__main__':
    expr()
    print("done")




    
