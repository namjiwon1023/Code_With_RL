import numpy as np
import torch as T
import os
from algorithm.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def choose_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = T.tensor(o, dtype=T.float32).unsqueeze(0).to(self.args.device)

            #########          must test        #####################
            pi, _ = self.policy.actor(inputs)
            u = pi.squeeze(0).detach().cpu().numpy()

        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

