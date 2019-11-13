
import numpy as np
from scipy.linalg import toeplitz
from typing import List

            
class LinearPolicy(object):
    
    def __init__(self, policy_params):
        
        self.init_seed = policy_params['seed']
        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
       
        self.w = self.weight_init(self.ob_dim * self.ac_dim, policy_params['zeros'])
        self.W = self.w.reshape(self.ac_dim, self.ob_dim)
        self.params = self.w

        self.N = len(self.params)
    
    def weight_init(self, d, zeros):
        
        if zeros:
            w = np.zeros(d)
        else:
            np.random.seed(self.init_seed)
            w = np.random.rand(d) / np.sqrt(d)
        return(w)
    
    def update(self, vec):        
        self.w += vec
        self.W = self.w.reshape(self.ac_dim, self.ob_dim)
        
    def forward(self, X):
        
        X = X.reshape(X.size, 1)

        return(np.tanh(np.dot(self.W, X)))
        
    def rollout(self, env, steps, incl_data=False, seed=0, train=True):
        env.seed(seed)
        state = env.reset()
        env._max_episode_steps = steps
        total_reward = 0
        done = False
        data=[]
        while not done:
            action = self.forward(state)
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
            action = action.reshape(len(action), )
            state, reward, done, _ = env.step(action)
            total_reward += reward
        return(total_reward)


class ToeplitzPolicy(object):
    
    def __init__(self, policy_params):
        
        self.init_seed = policy_params['seed']
        self.ob_dim = policy_params['ob_dim']
        self.h_dim = policy_params['h_dim']
        self.ac_dim = policy_params['ac_dim']
        
        self.w1 = self.weight_init(self.ob_dim + self.h_dim -1, policy_params['zeros'])
        self.w2 = self.weight_init(self.h_dim * 2 - 1, policy_params['zeros'])
        self.w3 = self.weight_init(self.ac_dim + self.h_dim - 1, policy_params['zeros'])
        
        self.W1 = self.build_layer(self.h_dim, self.ob_dim, self.w1)
        self.W2 = self.build_layer(self.h_dim, self.h_dim, self.w2)
        self.W3 = self.build_layer(self.ac_dim, self.h_dim, self.w3)
        
        self.b1 = self.weight_init(self.h_dim, policy_params['zeros'])
        self.b2 = self.weight_init(self.h_dim, policy_params['zeros'])
    
        self.params = np.concatenate([self.w1, self.b1, self.w2, self.b2, self.w3])
        self.N = len(self.params)
    
    def weight_init(self, d, zeros):
        if zeros:
            w = np.zeros(d)
        else:
            np.random.seed(self.init_seed)
            w = np.random.rand(d) / np.sqrt(d)
        return(w)
    
    def build_layer(self, d1, d2, v):
        # len v = d1 + d2 - 1
        col = v[:d1]
        row = v[(d1-1):]
        
        W = toeplitz(col, row)
        return(W)
    
    def update(self, vec):
        
        self.params += vec
        
        self.w1 += vec[:len(self.w1)]
        vec = vec[len(self.w1):]

        self.b1 += vec[:len(self.b1)]
        vec = vec[len(self.b1):]
        
        self.w2 += vec[:len(self.w2)]
        vec = vec[len(self.w2):]

        self.b2 += vec[:len(self.b2)]
        vec = vec[len(self.b2):]

        self.w3 += vec
        
        self.W1 = self.build_layer(self.h_dim, self.ob_dim, self.w1)
        self.W2 = self.build_layer(self.h_dim, self.h_dim, self.w2)
        self.W3 = self.build_layer(self.ac_dim, self.h_dim, self.w3)
        
    def forward(self, X):
        
        z1 = np.tanh(np.dot(self.W1, X) + self.b1)
        z2 = np.tanh(np.dot(self.W2, z1) + self.b2)
        return(np.tanh(np.dot(self.W3, z2)))
        
    def rollout(self, env, steps, incl_data=False, seed=0, train=True):
        env.seed(seed)
        state = env.reset()
        env._max_episode_steps = steps
        total_reward = 0
        done = False
        while not done:
            action = self.forward(state)
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
            action = action.reshape(len(action), )
            state, reward, done, _ = env.step(action)
            total_reward += reward
        return(total_reward)

