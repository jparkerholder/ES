
from optimizers import Adam
from utils import *
from shared_noise import *
from experiments import get_experiment

import ray
import gym
import parser
import argparse
import numpy as np
import pandas as pd
import os
import socket

@ray.remote
class Worker(object):  
            
    def __init__(self, env_seed, env_name='', deltas=None, delta_std=0.02):
        
        self.env = gym.make(env_name)
        self.env.seed(0)
        
        self.params = {}
        self.params['env_name'] = env_name
        self.params['ob_dim'] = self.env.observation_space.shape[0]
        self.params['ac_dim'] = self.env.action_space.shape[0]
        self.params = get_experiment(self.params)
                
        self.params['zeros'] = True
        self.params['seed'] = 0

        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.delta_std = delta_std

    def do_rollouts(self, policy, num_rollouts, seed=0, train=True):
        # initialize reward and delta arrays
        rollout_rewards, deltas_idx = [], []
        steps = 0
        
        for i in range(num_rollouts):
            
            self.policy = get_policy(self.params)
            idx, delta = self.deltas.get_delta(policy.size)
            delta = (self.delta_std * delta).reshape(self.policy.params.shape)
            deltas_idx.append(idx)

            self.policy.update(policy + delta)
            pos_reward, pos_steps = self.rollout(seed, train)
            
            self.policy.update(-2 * delta)
            neg_reward, neg_steps = self.rollout(seed, train)

            rollout_rewards.append([pos_reward, neg_reward])
            steps += pos_steps + neg_steps
        
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, 'steps' : steps}
    
    def rollout(self, seed=0, train=True):
        self.env.seed(seed)
        state = self.env.reset()
        self.env._max_episode_steps = self.params['steps']
        total_reward = 0
        done = False
        timesteps = 0
        while not done:
            action = self.policy.forward(state)
            action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
            action = action.reshape(len(action), )
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            timesteps += 1
        return(total_reward, timesteps)

def aggregate_rollouts(master, params):

    policy_id = ray.put(master.policy.params)
    num_rollouts = params['sensings']
    rollout_ids = [worker.do_rollouts.remote(policy_id, num_rollouts = num_rollouts) for worker in master.workers]
    results = ray.get(rollout_ids)
    
    rollout_rewards, deltas_idx = [], []
    timesteps = 0
    for result in results:
        deltas_idx += result['deltas_idx']
        rollout_rewards  += result['rollout_rewards']
        timesteps += result['steps']

    deltas_idx = np.array(deltas_idx)

    rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
    rollout_rewards = (rollout_rewards - np.mean(rollout_rewards)) / (np.std(rollout_rewards) +1e-8)

    g_hat, count = batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (master.deltas.get(idx, master.policy.params.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
    g_hat /= deltas_idx.size
    
    return(g_hat, timesteps)


class Learner(object):

    def __init__(self, params):

        params['zeros'] = False
        self.policy = get_policy(params)

        self.timesteps = 0
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = params['seed'] + 3)

        self.num_workers = params['num_workers']
        self.workers = [Worker.remote(params['seed'] + 7 * i,
                                      env_name=params['env_name'],
                                      deltas= deltas_id,
                                      delta_std=params['sigma']) for i in range(params['num_workers'])]

def train(params):
        
    env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]

    m, v = 0, 0
        
    master = Learner(params)
        
    n_eps = 0
    n_iter = 0
    ts_cumulative = 0
    ts, rollouts, rewards = [], [], []
        
    while n_iter < params['max_iter']:
        
        reward = master.policy.rollout(env, params['steps'])
        rewards.append(reward)
        rollouts.append(n_eps)
        ts.append(ts_cumulative)
        
        print('Iter: %s, Eps: %s, R: %s' %(n_iter, n_eps, np.round(reward,4)))
            
        params['n_iter'] = n_iter
        gradient, timesteps = aggregate_rollouts(master, params)
        ts_cumulative += timesteps
        n_eps += 2 * params['sensings']

        gradient /= (np.linalg.norm(gradient) / master.policy.N + 1e-8)
        
        n_iter += 1
        update, m, v = Adam(gradient, m, v, params['learning_rate'], n_iter)
            
        master.policy.update(update)

        out = pd.DataFrame({'Rollouts': rollouts, 'Reward': rewards, 'Timesteps': ts})
        out.to_csv('data/%s/results/%s_Seed%s.csv' %(params['dir'], params['filename'], params['seed']), index=False) 

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Swimmer-v2')
    parser.add_argument('--seed', '-sd', type=int, default=0)
    parser.add_argument('--max_iter', '-it', type=int, default=200)
    parser.add_argument('--num_workers', '-nw', type=int, default=4)

    parser.add_argument('--filename', '-f', type=str, default='') # anything else you want to add

    args = parser.parse_args()
    params = vars(args)
    params = get_experiment(params)

    params['dir'] = params['env_name'] + '_' + params['filename']

    if not(os.path.exists('data/'+params['dir'])):
        os.makedirs('data/'+params['dir'])
        os.makedirs('data/'+params['dir']+'/results')

    ray.init()

    train(params)

if __name__ == '__main__':
    main()

