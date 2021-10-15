import torch as T
import torch.nn as nn
import numpy as np
import random
import ray
import gym
import time
import os

class Dict(dict):
    def __init__(self,config, section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)

def _random_seed(env, seed):
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
