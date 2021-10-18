import torch
import numpy as np
from utils.environment import Environment 

def run_setting(args, agent_args):
    env = Environment(args.env_name)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent_args['discrete'] = env.is_discrete

    ##tensorboard
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    else :
        writer = None

    ##device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.use_cuda == False:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    return args, agent_args, env, state_dim, action_dim, writer, device

def make_transition(state, action, reward, next_state, done, log_prob):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['done'] = done
    transition['log_prob'] = log_prob
    return transition

def convert_to_tensor(*value):
    device = value[0]
    return [torch.tensor(x).float().to(device) for x in value[1:]]

class Dict(dict):
    def __init__(self,config,section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)