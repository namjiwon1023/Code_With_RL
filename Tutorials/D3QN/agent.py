import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import copy
import gym
import time

from network import QNetwork
from ReplayBuffer import ReplayBuffer

class D3QNAgent(object):
    def __init__(self, args):

        self.args = args
        self.epsilon = 1.0

        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.eval = QNetwork(self.n_states, self.n_actions, args)

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


        if os.path.exists(self.model_path + '/D3QN.pth'):
            self.load_models()


        self.total_step = 0

    def choose_action(self, state):

        s = T.FloatTensor(state).to(self.args.device)

        if self.epsilon >= np.random.random() and not self.args.evaluate:
            choose_action = np.random.randint(0, self.n_actions)
        elif self.args.evaluate:
            choose_action = self.eval(s).argmax()
            choose_action = choose_action.detach().cpu().numpy()
        else :
            choose_action = self.eval(s).argmax()
            choose_action = choose_action.detach().cpu().numpy()

        if not self.args.evaluate:
            self.transition = [state, choose_action]
        return choose_action

    def target_net_update(self):
        self.target.load_state_dict(self.eval.state_dict())

    def soft_target_net_update(self):
        for t_p, l_p in zip(self.target.parameters(), self.eval.parameters()):
            t_p.data.copy_((1 - self.args.tau) * t_p.data + self.args.tau * l_p.data)


    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            next_q = self.target(next_state).gather(1, self.eval(next_state).argmax(dim = 1, keepdim = True))

        curr_q = self.eval(state).gather(1, action)

        target_q = reward + next_q * mask

        loss = (target_q - curr_q)**2
        loss = loss.mean()

        self.eval.optimizer.zero_grad()
        loss.backward()
        self.eval.optimizer.step()

        if self.total_step % self.args.update_rate == 0:
            self.target_net_update()
            # self.soft_target_net_update()
        return loss.item()

    def evaluate_agent(self, n_starts=10):
        reward_sum = 0
        for _ in range(n_starts):
            done = False
            state = self.env.reset()
            while (not done):
                if self.args.evaluate:
                    self.env.render()
                # time.sleep(0.02)
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                state = next_state
        # self.env.close()
        return reward_sum / n_starts

    def save_models(self):
        print('------ Save models ------')
        self.eval.save_model()

    def load_models(self):
        print('------ load models ------')
        self.eval.load_model()
