### Experiments ###

import learners
import specs
import envs
import os
import pickle
import copy
from time import time
import random
import numpy as np
import utils
import itertools
from csv import reader
import sys
import torch
from torch import tensor as tt

from environments.test_envs import mg0

# Debugging
test_hps = {'actual_dist': True,
            'augment_data': False,
            'buffers': { 'actors': {'size': 1000, 'batch': 32},
                         'critics': {'size': 1000, 'batch': 32} },
            'continue_prob': 0.9,
            'entropy_weight': 100.0,
            'epsilon': 0.00,
            'gamma_Phi' : 0.99,
            'kl_target' : 0.025,
            'l2_weight' : 0.0001,
            # 'learning_rates': { 'actors': (('constant', 0.0001),),
            #                     'critics': ('constant', 0.001),
            #                     'lagrange_multiplier': (('constant', 0.005),) },
            'learning_rates': { 'actors': (('constant', 1.0),),
                                'critics': ('constant', 1.0),
                                'lagrange_multiplier': (('constant', 1.0),) },
            'models': { 'actors': {'type':'dnn', 'shape':[64,64]},
                        'critics': {'type':'dnn', 'shape':[64,64]} },
            'normalise_advantages' : True,
            'num_updates' : { 'actors': None,
                              'critics': None },
            'optimisers': { 'actors': 'adam',
                            'critics': 'adam' },
            'patient_updates': True,
            'sequential': False,
            'spec_reward': 10,
            'until_converged' : { 'actors': True,
                                  'critics': False },
            'update_after': { 'actors': 100,
                              'critics': 10 }}


def oc_test(root=os.getcwd(), max_steps=1000000, hps=test_hps, repetitions=1):

    location = '{}/experiments/oc_test'.format(root)

    env = envs.EnvWrapper('oc_test_game', 'overcooked', envs.OvercookedGame(envs.oc0.map, envs.oc0.recipes))

    # All
    # specifications = ('(F G phi) | (F G psi)', 'F (psi & X phi)')
    # reward_functions = ({'reward':mg0.reward_1, 'discount':0.8}, {'reward':mg0.reward_2, 'discount':0.9})
    # objectives = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

    # Minimal spec
    specifications = ('F (tomatopickup0 | tomatopickup1)',)
    # specifications = ('(G psi) | (G phi)',)
    # specifications = ('(G phi) | (G psi)',)
    reward_functions = ()
    objectives = (np.array((1.0,)),)
    # objectives = (np.array((1.0,0.0)),np.array((0.0,1.0)))

    spec_controller = specs.Spec_Controller(specifications, load_from=root)
    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    learner = make_learner('almanac', obs_size, act_sizes, len(objectives[0]), len(objectives), hps)
    prefix = "{}-{}-{}-{}".format(env.name, learner.name, 0, 0)

    for _ in range(repetitions):

        # Create learner
        run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000, verbose=True)
    
    # run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000, evaluate=True)




def debug(root=os.getcwd(), max_steps=100000, hps=test_hps, repetitions=1):

    location = '{}/experiments/debug'.format(root)

    env = envs.EnvWrapper('debug_game', 'mg', envs.mg_0)

    all_states = tt([[[1,0,1,0,0,0], # (0,0)
                    [1,0,0,1,0,0],   # (0,1)
                    [1,0,0,0,1,0],   # (0,2)
                    [1,0,0,0,0,1]],  # (0,3)
                    [[0,1,1,0,0,0],  # (1,0)
                    [0,1,0,1,0,0],   # (1,1)
                    [0,1,0,0,1,0],   # (1,2)
                    [0,1,0,0,0,1]]]) # (1,3)

    # All
    # specifications = ('(F G phi) | (F G psi)', 'F (psi & X phi)')
    # reward_functions = ({'reward':mg0.reward_1, 'discount':0.8}, {'reward':mg0.reward_2, 'discount':0.9})
    # objectives = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

    # Minimal spec
    specifications = ('G (phi -> X psi)','G F phi')
    # specifications = ('(G psi) | (G phi)',)
    # specifications = ('(G phi) | (G psi)',)
    reward_functions = ()
    objectives = (np.array((1.0,1.0)),)
    objectives = utils.normalise_objs(objectives)
    # objectives = (np.array((1.0,0.0)),np.array((0.0,1.0)))

    spec_controller = specs.Spec_Controller(specifications, load_from=root)
    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    learner = make_learner('almanac', obs_size, act_sizes, len(objectives[0]), len(objectives), hps)
    prefix = "{}-{}-{}-{}".format(env.name, learner.name, 0, 0)

    for _ in range(repetitions):

        # Create learner
        run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000)
    
    run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000, evaluate=True)


# Experiment 0
exp0_hps = {'actual_dist': True,
            'augment_data': False,
            'buffers': { 'actors': {'size': 1000, 'batch': 32},
                         'critics': {'size': 1000, 'batch': 32} },
            'continue_prob': 0.9,
            'epsilon': 0,
            'gamma_Phi' : 0.99,
            'kl_target' : 0.025,
            'learning_rates': { 'actors': (('constant', 0.001),),
                                'critics': ('constant', 0.01),
                                'lagrange_multiplier': (('constant', 0.01),) },
            'models': { 'actors': {'type':'dnn', 'shape':[12]},
                        'critics': {'type':'dnn', 'shape':[12]} },
            'optimisers': { 'actors': 'sgd',
                            'critics': 'sgd' },
            'patient_updates': True,
            'spec_reward': 10,
            'update_after': { 'actors': 1000,
                              'critics': 1000 }}

def exp0(root, id, max_steps, hps=exp0_hps, repetitions=10):

    location = '{}/experiments/0'.format(root)

    if id["run"] == 1:
        env = envs.mg_1
        specifications = ('G (phi & psi)', '(!chi) U phi')
        reward_functions = ()
        objectives = ()
    elif id["run"] == 2:
        env = envs.mg_2
        specifications = ('G !chi', 'G F phi', 'F psi')
        reward_functions = ()
        objectives = ()
    elif id["run"] == 3:
        env = envs.mg_3
        specifications = ('F G psi', 'F phi')
        reward_functions = ()
        objectives = ()
    else:
        pass

    spec_controller = specs.Spec_Controller(specifications, load_from=root)
    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    for r in range(repetitions):

        # Create learner
        learner = make_learner('almanac', obs_size, act_sizes, len(specifications), len(objectives), hps)
        prefix = "{}-{}-{}-{}".format(env.name, learner.name, id["run"], r)

        run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000)

# Experiment 1
exp1_hps = {'actual_dist': True,
            'augment_data': True,
            'buffers': { 'actors': {'size': 1000, 'batch': 32},
                         'critics': {'size': 1000, 'batch': 32} },
            'continue_prob': 0.9,
            'epsilon': 0,
            'gamma_Phi' : 0.99,
            'kl_target' : 0.025,
            'learning_rates': { 'actors': (('constant', 0.001),),
                                'critics': ('constant', 0.75),
                                'lagrange_multiplier': (('constant', 0.01),) },
            'models': { 'actors': {'type':'dnn', 'shape':[24, 32, 24]},
                        'critics': {'type':'dnn', 'shape':[24, 24, 24]} },
            'optimisers': { 'actors': 'sgd',
                            'critics': 'sgd' },
            'patient_updates': True,
            'spec_reward': 10,
            'update_after': { 'actors': 1000,
                              'critics': 1000 }}

def exp1(num_specs, num_actors, num_states, num_run, root, id, max_steps, hps=exp1_hps):

    # location = './{}/experiments/{}'.format(root, id["num"])
    location = '{}/experiments/1'.format(root)
    name = '{}-{}-{}-{}'.format(num_states, num_actors, num_specs, num_run)

    # Specifications
    labels = ['l{}'.format(i) for i in range(num_specs)]
    possible_specs = [  lambda x : 'F {}'.format(x),\
                        lambda x : 'G {}'.format(x),\
                        lambda x : 'F G {}'.format(x),\
                        lambda x : 'G F {}'.format(x),\
                        lambda x : 'X (X {})'.format(x),\
                        lambda x, y : '{} U {}'.format(x, y),\
                        lambda x, y : 'F ({} & {})'.format(x, y),\
                        lambda x, y : 'G ({} | (X {}))'.format(x, y),\
                        lambda x, y : 'F G ({} | {})'.format(x, y),\
                        lambda x, y : 'G F ({} & (X {}))'.format(x, y)  ]
    possible_weights = [2, 5, 8]
    if num_states == 1:
        possible_specs = possible_specs[:5]
    
    # Map parameters
    state_size = num_states
    action_sizes = num_actors * [2]
    num_rules = max(1, int(0.5 * state_size))
    num_antecedents = max(1, int(0.5 * state_size))

    completed = False
    while not completed:

        # Form objectives
        specifications = random.sample(possible_specs, num_specs)
        weights = random.sample(possible_weights, num_specs)
        sum_weights = sum(weights)
        objectives = [tuple(w / sum_weights for w in weights)]

        # Form env
        smg = envs.StructuredMarkovGame(state_size, action_sizes, num_rules, num_antecedents)
        env = envs.EnvWrapper('smg-{}-{}'.format(id['run'], num_run), 'smg', smg)

        # Form input parameters and LDBAs
        spec_controller = specs.Spec_Controller(specifications, load_from=root)
        obs_size = env.get_obs_size() + sum(spec_controller.num_states)
        act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

        # Evaluate using PRISM
        spec_controller.save_props(location, name, objectives[0])
        env.model.create_prism_model(num, spec_controller.specs, location)

        # Sometimes the MMG and spec combination generated is trivial, if so we retry
        

        def run_prism(location, name, cuddmaxmem=16,javamaxmem=16, epsilon=0.001, maxiters=100000, timeout=72000):

            run_name = 'prism -cuddmaxmem {}g -javamaxmem {}g -epsilon {} -maxiters {} -timeout {} '.format(cuddmaxmem, javamaxmem, epsilon, maxiters, timeout)
            model_name = '{}/prism_models/{}.prism '.format(location, name)
            props_name = '{}/prism_specs/{}.props -prop 1 '.format(location, name)
            save_name = '> {}/prism_evaluations/{}-true.txt'.format(location, name)
            t0 = time()
            os(run_name + model_name + props_name + save_name)
            t1 = time()
            time = t1 - t0
            

            return prob, time
        
        prob, time = run_prism()
        
        if prob == 0.0:
            continue
        else:
            prism_











            completed = True
            

        prism -cuddmaxmem 16g -javamaxmem 16g -epsilon 0.001 -maxiters 100000 -timeout 72000 /home/hert5888/almanac/experiments/1/prism_models/1-1-1-2.prism /home/hert5888/almanac/experiments/1/prism_specs/1-1-1-2.props -prop 1 > /home/hert5888/almanac/experiments/1/prism_evaluations/1-1-1-2-true.txt

        call_mifuntion_vers_1()
        t1 = time()
        call_mifunction_vers_2()
        t2 = time()





        run(learner, env, 10000, spec_controller, [], objectives, location, prefix, num_plot_points=1000)
        with open('{}/scores/{}-scores.txt'.format(location, prefix), 'r') as f:
            data = list(reader(f))
            if np.var(np.array(d[0] for d in data[10:60])) > 0.01:
                completed = True
        if completed:
            run(learner, env, max_steps - 10000, spec_controller, [], objectives, location, prefix)
        else:
            continue



        # Create learner
        learner = make_learner('almanac', obs_size, act_sizes, len(specifications), len(objectives), hps)
        prefix = "{}-{}-{}-{}".format(env.name, learner.name, id["run"], num_run)

        # Sometimes the MMG and spec combination generated is trivial, if so we retry
        run(learner, env, 10000, spec_controller, [], objectives, location, prefix, num_plot_points=1000)
        with open('{}/scores/{}-scores.txt'.format(location, prefix), 'r') as f:
            data = list(reader(f))
            if np.var(np.array(d[0] for d in data[10:60])) > 0.01:
                completed = True
        if completed:
            run(learner, env, max_steps - 10000, spec_controller, [], objectives, location, prefix)
        else:
            continue

        # Save specs and weights

        # Save policy distributions
        possible_game_states = list(itertools.product([0, 1], repeat=env.get_obs_size()))
        possible_spec_states = [[tuple((1 if i == j else 0) for i in range(s_s)) for j in range(s_s)] for s_s in spec_controller.num_states]
        possible_states = [utils.flatten(p_s) for p_s in itertools.product(possible_game_states, *possible_spec_states)]
        possible_state_tensors = torch.stack([tt(p_s).float() for p_s in possible_states])
        p_dists = learner.get_policy_dists(possible_states, possible_state_tensors)
        
        # Create PRISM models
        num = '{}-{}'.format(id["run"], r)
        env.model.create_prism_model(num, spec_controller.specs, location)
        env.model.create_prism_model(num, spec_controller.specs, location, policy=p_dists)
        env.model.create_prism_model(num, spec_controller.specs, location, policy=p_dists, det=True)

    # Save information about objectives
    # spec_controller.save_model(location, env.name, id["run"])
    # # with open("{}/{}-{}-reward_functions.pickle".format(location, env.name, id["run"]), 'wb') as f:
    # #     pickle.dump(reward_functions, f)
    # with open("{}/objectives/{}-{}-objectives.pickle".format(location, env.name, id["run"]), 'wb') as f:
    #     pickle.dump(objectives, f)


def old_exp1(state_size, num_actors, num_specs, run_num):

    completed = False
    while not completed:

        specifications = random.sample(possible_specs, num_specs)
        weights = random.sample(possible_weights, num_specs)
        sum_weights = sum(weights)
        weights = [w / sum_weights for w in weights]
        action_sizes = [random.randint(2,4) for r in range(num_actors)]
        mmg = envs.MatrixMarkovGame(state_size, action_sizes, labels, sparsity=0.6, structured_labels=True, nonlinearities=0.4)
        
        mmg_hps['env'] = mmg
        mmg_hps['env_name'] = None
        mmg_hps['labeller'] = mmg.labeller
        mmg_hps['model_id'] = None
        mmg_hps['specs'] = list(zip(specifications, weights))

        # filename = 'results/experiment_1/scores/almanac-{}-{}-{}-{}.txt'.format(state_size, num_actors, num_specs, run_num)
        # modelname = 'results/models/mmg/almanac-{}-{}-{}-{}.pickle'.format(state_size, num_actors, num_specs, run_num)
        # with open(filename, 'w') as f:
        #     f.write("State size: {}\n".format(state_size))
        #     a_s_line = ', '.join([str(a) for a in action_sizes])
        #     f.write("Action sizes: {}\n".format(a_s_line))
        #     s_line = ', '.join(['{}: {}'.format(w, s) for s, w in zip(specifications, weights)])
        #     f.write("Specs: {}\n".format(s_line))
        #     f.write("Run: {}\n\n".format(run_num))

        specs_name = 'results/experiment_1/specs/matrix_markov_games/{}-{}-{}-{}.props'.format(state_size, num_actors, num_specs, run_num)
        with open(specs_name, 'w') as f:
            if num_specs == 1:
                f.write('Pmax=? [ X ( ' + specifications[0] + ' ) ]\n\n')
                f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
            else:
                f.write('multi( Pmax=? [ X ( ' + specifications[0] + ' ) ] , Pmax=? [ X ( ' + specifications[1] + ' ) ] )\n\n')
                f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
                f.write('P=? [ X ( ' + specifications[1] + ' ) ]\n\n')

        weights_name = 'results/experiment_1/specs/matrix_markov_games/{}-{}-{}-{}.weights'.format(state_size, num_actors, num_specs, run_num)
        with open(weights_name, 'w') as f:
            for w in weights:
                f.write('{}\n'.format(w))

        completed = run(mmg_hps, run_num, filename)

        possible_game_states = list(itertools.product([0, 1], repeat=env.get_obs_size()))
        possible_spec_states = []
        for s_s in self.ldba_state_sizes:
            spec_states = [[0 for i in range(s_s)] for j in range(s_s)]
            for k in range(s_s):
                spec_states[k][k] = 1
            possible_spec_states.append([tuple(s_s) for s_s in spec_states])
        possible_product_states = itertools.product(possible_game_states, *possible_spec_states)
        self.possible_states = [tuple(utils.flatten(p_s)) for p_s in possible_product_states]
        self.possible_state_tensors = torch.stack([tt(p_s).float() for p_s in self.possible_states])
        self.policy_dists = None

        if s % score_interval == 0:

            # Sometimes the mmg generated is trivial, if so we return false and regenerate the mmg
            if self.env_kind == 'mmg':
                if len(first_50) < 60:
                    first_50.append(recent_game_score/score_interval)
                else:
                    if not on2awinner:
                        if np.var(first_50[10:]) < 0.01:
                            return False
                        else:
                            on2awinner = True
            
            last_score_interval.append(recent_game_score / score_interval)

            with open(filename, 'a') as f:
                f.write('{}, {}, {}\n'.format(int(s / score_interval), average_game_score, recent_game_score / score_interval))
            recent_game_score = 0

            if s > (steps / 10) and average_game_score > best + 0.05:
                if self.env_kind == 'mmg':
                    self.policy_dists = dict([(p, self.policy(t, probs=True)) for p,t in zip(self.possible_states, self.possible_state_tensors)])
                best = average_game_score

        if len(last_score_interval) == score_interval:
            if np.var(last_score_interval) < 0.001:
                average_game_score = game_score / s
                if average_game_score > best + 0.05:
                    if self.env_kind == 'mmg':
                        self.policy_dists = dict([(p, self.policy(t, probs=True)) for p,t in zip(self.possible_states, self.possible_state_tensors)])
                    best = average_game_score
                s = steps + 1


# Experiment 2


# Experiment 3


# Run experiment instance
def old_run(hp, num, filename=None, modelname=None):

    env = envs.EnvWrapper(hp['env_name'], hp['env_type'], hp['env'], hp['labeller'])
    spec = []
    for s in hp['specs']:
        f = "specs/" + s[0] + '.pickle'
        if os.path.isfile(f):
            old_spec = pickle.load(open(f, "rb"))
            spec.append(old_spec)
        else:
            new_spec = specs.Spec(s[0], s[1])
            new_spec.save()
            spec.append(new_spec)
    model_constants = { 'discounts': hp['discounts'],
                        'l2_reg': hp['l2_regulariser_weight'],
                        'lrs': hp['learning_rates'] }
    almanac = learners.Almanac(env, spec, hp['optimisers'], hp['buffers'], hp['models'], hp['local'], model_constants, hp['model_id'])
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
    trained = almanac.train(hp['steps'], env, spec, hp['actual_state_dist'], hp['patient_updates'], train_constants, hp['run_id'], filename)
    if hp['env_type'] == 'mmg':
        env.model.create_prism_model(num, spec)
        env.model.create_prism_model(num, spec, policy=almanac.get_policy_dists())
        env.model.create_prism_model(num, spec, policy=almanac.get_policy_dists(), det=True)
    if modelname != None:
        almanac.lrs = None
        almanac.patient_buffer = None
        almanac.hasty_buffer = None
        with open(modelname, 'wb') as f:
            pickle.dump(almanac, f)
    
    return trained

def make_learner(learner_name, obs_size, act_sizes, num_rewards, num_objectives, hps):

    if learner_name == 'almanac':
        learner = learners.Almanac(obs_size, act_sizes, num_rewards, num_objectives, hps)
    else:
        print("Error: Agent name must be \'almanac\' or")
        return

    return learner


def experiment(root, id, env, formulae, max_steps, reward_functions, objectives, learner_name, repetitions=1):

    # Form input parameters and LDBAs
    location = '{}/experiments/{}'.format(root, id["num"])
    spec_controller = specs.Spec_Controller(formulae)
    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    # Run experiment
    for r in range(repetitions):

        learner = make_learner(learner_name, obs_size, act_sizes)
        prefix = "{}-{}-{}-{}".format(env.name, learner.name, id["run"], r)
        run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix)

    # Save information about objectives
    spec_controller.save_model(location, env.name, id["run"])
    with open("{}/{}-{}-reward_functions.pickle".format(location, env.name, id["run"]), 'wb') as f:
        pickle.dump(reward_functions, f)
    with open("{}/objectives/{}-{}-objectives.pickle".format(location, env.name, id["run"]), 'wb') as f:
        pickle.dump(objectives, f)


def run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000, evaluate=False, verbose=False):

    # Check for stupid mistakes
    num_specs = len(spec_controller.specs)
    num_reward_functions = len(reward_functions)
    num_rewards = num_specs + num_reward_functions
    num_objectives = len(objectives)
    for o in objectives:
        if len(o) != num_specs + num_reward_functions:
            print('Error: Objectives must contain weights for every spec and reward function')
            return
        if sum(o[:num_reward_functions]) > 0 and sum(o[num_reward_functions:]) > 0:
            print('Error: Objectives must combine only specs or only reward functions')
            return
    if len(learner.lrs['actors']) != num_objectives or len(learner.lrs['lagrange_multiplier']) != num_objectives:
        print('Error: Must be the same number of learning rates for actors and Lagrange multipliers as objectives')
        return

    # Prepare for saving scores
    score_interval = int(max_steps / num_plot_points)
    recent_scores = [0.0 for _ in objectives]
    total_scores = [0.0 for _ in objectives]
    postfix = "-eval" if evaluate else ""
    with open('{}/scores/{}-scores{}.txt'.format(location, prefix, postfix), 'w') as f:
        obj_list = [str(i) for i in range(num_objectives)]
        f.write("recent_" + ",recent_".join(obj_list) + ",total_" + ",total_".join(obj_list) + "\n")

    # Avoid duplicating the computation of automaton transitions
    if learner.hps['augment_data']:
        transitions = dict()

    s = 0
    while s < max_steps:

        # Initialise environment
        game_state = env.reset()
        spec_states, acceptances = spec_controller.reset() 
        done = False
        t = 0
        current_discounts = np.ones(num_rewards)
        # print("End of episode, resetting...")

        while not done and s < max_steps:

            # Form state vectors 
            f_game_state = env.featurise(game_state)
            f_spec_states = spec_controller.featurise(spec_states)
            prod_state = torch.cat([f_game_state] + f_spec_states, 0)
            
            # Perform joint action
            joint_action = learner.act(prod_state)

            # If an epsilon transition is made the game (and possibly spec) state remains the same
            e_ts = spec_controller.is_epsilon_transition(joint_action, env.get_act_sizes())
            if len(e_ts) > 0:
                is_e_t = True
                new_game_state, done = game_state, done
            else:
                is_e_t = False
                new_game_state, done = env.step(joint_action)
            

            if verbose:
                print(env.model.overcooked)


            # Update LDBAs
            label_set = env.label(new_game_state)
            new_spec_states, acceptances = spec_controller.step(e_ts, label_set)

            for k in env.model.labels.keys():
                for a in [0,1]:
                    if env.model.labels[k][a]:
                        print("Agent {} did {}".format(a, k))
                        print("hurrah")

            # if env.model.labels['tomato_pickup'][0] or env.model.labels['tomato_pickup'][1]:
            #     print("HYPE")

            held = False
            if new_game_state.players[0].held_object != None and game_state.players[0].held_object == None:
                print("Agent 0 is holding {}".format(new_game_state.players[0].held_object))
                held = True
            if new_game_state.players[1].held_object != None and game_state.players[1].held_object == None:
                print("Agent 1 is holding {}".format(new_game_state.players[1].held_object))
                held = True

            if held:
                print("woah")

            if label_set != []:
                print(label_set)              
                print("wahey!")


            # Form new state vectors
            f_new_game_state = env.featurise(new_game_state)
            f_new_spec_states = spec_controller.featurise(new_spec_states)

            if random.random() > learner.hps['continue_prob']:
                done = True

            if not evaluate:

                # Form learning input
                info = {"s": game_state,
                        "f(s)": f_game_state,
                        "q": spec_states,
                        "f(q)": f_spec_states,
                        "a": joint_action,
                        "s'": new_game_state,
                        "f(s')": f_new_game_state,
                        "q'": new_spec_states,
                        "f(q')": f_new_spec_states,
                        "R": reward_functions,
                        "F": acceptances,
                        "t": t,
                        "o_s": objectives,
                        "D": done,
                        "is_e_t": is_e_t}
                if learner.hps['augment_data']:
                    if (e_ts, label_set) not in transitions.keys():
                        transitions[(e_ts, label_set)] = spec_controller.get_transitions(e_ts, label_set)
                    info["f(q)_s"], info["f(q')_s"], info["F_s"] = transitions[(e_ts, label_set)]
                else:
                    info["f(q)_s"], info["f(q')_s"], info["F_s"] = [f_spec_states], [f_new_spec_states], [acceptances]
                
                # Update learners
                learner.step(info, objectives)
                
            # Record Almanac-esque scores
            discounts = []
            rewards = []
            for j in range(num_specs):
                discounts.append(learner.hps['gamma_Phi'] if acceptances[j] else 1.0)
                rewards.append(learner.hps['spec_reward'] if acceptances[j] else 0.0)
            for j in range(num_specs,num_rewards):
                discounts.append(reward_functions[j]['discount'] if acceptances[j] else 1.0)
                rewards.append(reward_functions[j]['reward'] if acceptances[j] else 0.0)
            current_discounts *= np.array(discounts)
            discounted_rewards = current_discounts * rewards
            # utilities = [sum(z[0] * z[1] for z in zip(o, discounted_rewards)) for o in objectives]
            utilities = [sum(discounted_rewards * o) for o in objectives]
            recent_scores = [recent_scores[i] + utilities[i] for i in range(num_objectives)]

            # Update variables for next step
            game_state = new_game_state
            spec_states = new_spec_states
            t += 1
            s += 1

            # Save and occasionally print (average) score
            if s % score_interval == 0:
                total_scores = [total_scores[i] + recent_scores[i] for i in range(num_objectives)]
                rs_s = ",".join(["{:.5f}".format(rs / score_interval) for rs in recent_scores])
                ts_s = ",".join(["{:.5f}".format(ts / s) for ts in total_scores])
                print("Average Score ({}/{}): {} (recent)   {} (total)".format(int(s / score_interval), num_plot_points, rs_s, ts_s))
                with open('{}/scores/{}-scores{}.txt'.format(location, prefix, postfix), 'a') as f:
                    f.write(rs_s + ',' + ts_s + '\n')
                recent_scores = [0.0 for _ in objectives]

    # Save model
    if not evaluate:
        learner.save_model(location, prefix)

    return

# debug()
oc_test()