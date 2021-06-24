import copy
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym.wrappers import RescaleAction
import os

from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
from utils.ReplayBuffer import ReplayBuffer
from utils.noise import OUNoise

class DDPGAgent(object):
    def __init__(self, args):
        self.args = args
        # self.epsilon = 1.0

        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]


        self.noise = OUNoise(self.n_actions, theta=self.args.ou_noise_theta, sigma=self.args.ou_noise_sigma,)

        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        self.actor_eval = ActorNetwork(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticNetwork(self.n_states, self.n_actions, self.args)

        self.actor_target = copy.deepcopy(self.actor_eval)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/DDPG_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state):
        if self.total_step < self.args.start_step and not self.args.evaluate:
            # choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            choose_action = self.env.action_space.sample()
            # print('epsilon action :',choose_action)
        else :
            choose_action = self.actor_eval(T.FloatTensor(state).to(self.actor_eval.device)).detach().cpu().numpy()
            # print('train action : ', choose_action)

        if not self.args.evaluate:
            if self.args.Gaussian_noise:
                noise = np.random.normal(0, self.max_action*self.args.exploration_noise, size=self.n_actions)
                # print('Gaussian noise : ',noise)
            else:
                noise = self.noise.sample()
                # print('OU noise : ',noise)
            choose_action = np.clip(choose_action + noise, -1, 1)
            # print('clip action :', choose_action)
        self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action = self.actor_target(next_state)
            next_value = self.critic_target(next_state, next_action)
            target_values = reward + next_value * mask

        eval_values = self.critic_eval(state, action)
        critic_loss = self.critic_eval.loss_func(eval_values, target_values)

        self.critic_eval.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval.optimizer.step()

        # for p in self.critic_eval.parameters():
        #         p.requires_grad = False

        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()

        self.actor_eval.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_eval.optimizer.step()

        # for p in self.critic_eval.parameters():
        #         p.requires_grad = True

        if self.total_step % self.args.target_update_interval == 0:
            self.target_soft_update()

        return critic_loss.item(), actor_loss.item()

    def target_soft_update(self):
        with T.no_grad():
            for t_p, l_p in zip(self.actor_target.parameters(), self.actor_eval.parameters()):
                t_p.data.copy_(self.args.tau * l_p.data + (1 - self.args.tau) * t_p.data)
            for t_p, l_p in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
                t_p.data.copy_(self.args.tau * l_p.data + (1 - self.args.tau) * t_p.data)


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
        self.actor_eval.save_model()
        self.critic_eval.save_model()

    def load_models(self):
        print('------ load models ------')
        self.actor_eval.load_model()
        self.critic_eval.load_model()
