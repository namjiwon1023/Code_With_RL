import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction

from network import Actor, QNet

from replaybuffer import ReplayBuffer
from utils import disable_gradients, network_update, _save_model, _load_model, _target_soft_update, get_batch_buffer, mbpo_target_entropy_dict, get_probabilistic_num_min

class REDQSACAgent:
    def __init__(self, args):
        self.args = args

        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.names = locals()
        for i in range(self.args.num_Q):
            self.names[f'q_net_{i}_path'] = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic + '_%d' % i)

        # Environment setting
        self.env = gym.make(args.env_name)
        self.test_env = gym.make(args.env_name)

        self.env = RescaleAction(self.env, -1, 1)
        self.test_env = RescaleAction(self.test_env, -1, 1)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.init_random_steps = args.init_random_steps
        self.delay_update_steps = self.init_random_steps if args.delay_update_steps == 'auto' else args.delay_update_steps

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        self.criterion = nn.MSELoss()

        # actor-critic net setting
        self.actor = Actor(self.n_states, self.n_actions, self.args)

        # Collection of Q Network
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(self.args.num_Q):
            new_q_net = QNet(self.n_states, self.n_actions, self.args)
            self.q_net_list.append(new_q_net)
            new_q_target_net = copy.deepcopy(new_q_net)
            disable_gradients(new_q_target_net)
            self.q_target_net_list.append(new_q_target_net)

        # Temperature Coefficient
        if args.target_entropy == 'auto':
            self.target_entropy = -self.n_actions
        if args.target_entropy == 'mbpo':
            self.target_entropy = mbpo_target_entropy_dict[args.env_name]
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)
        self.q_optimizer_list = []
        for q_i in range(self.args.num_Q):
            self.q_optimizer_list.append(optim.Adam(self.q_net_list[q_i].parameters(), lr=self.args.critic_lr))

        self.total_step = 0

        self.critic_learning_step, self.actor_learning_step = 0, 0

    def select_exploration_action(self, state):
        with T.no_grad():
            if self._get_current_num_data() <= self.init_random_steps:
                select_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                select_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device), evaluate=False, with_logprob=True)
                select_action = select_action.detach().cpu().numpy()
            self.transition = [state, select_action]
        return select_action

    def select_test_action(self, state):
        with T.no_grad():
            select_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device), evaluate=True, with_logprob=False)
            select_action = select_action.detach().cpu().numpy()
        return select_action

    def _get_current_num_data(self):
        return len(self.memory)

    def learn(self, writer):
        num_update = 0 if self._get_current_num_data() <= self.delay_update_steps else self.args.utd_ratio
        for i_update in range(num_update):
            self.critic_learning_step += 1
            state, next_state, action, reward, mask = get_batch_buffer(self.memory, self.args.batch_size, self.args.device, self.n_actions)
            # TD error
            # update value
            critic_loss = self._value_update(state, next_state, action, reward, mask)

            for q_i in range(self.args.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            critic_loss.backward()
            for q_i in range(self.args.num_Q):
                self.q_optimizer_list[q_i].step()

            for q_i in range(self.args.num_Q):
                for p in self.q_net_list[q_i].parameters():
                    p.requires_grad = False

            if ((i_update + 1) % self.args.policy_update_delay == 0) or i_update == num_update - 1:
                self.actor_learning_step += 1
                actor_loss, new_log_prob = self._policy_update(state)

                # update Policy
                network_update(self.actor_optimizer, actor_loss)

                # update Temperature Coefficient
                alpha_loss = self._temperature_update(new_log_prob)

                network_update(self.alpha_optimizer, alpha_loss)

                self.alpha = self.log_alpha.exp()

                if self.actor_learning_step % 1000 == 0:
                    writer.add_scalar("loss/actor", actor_loss.detach().item(), self.actor_learning_step)
                    writer.add_scalar("loss/alpha", alpha_loss.detach().item(), self.actor_learning_step)
                    writer.add_scalar("value/alpha", self.alpha.detach().item(), self.actor_learning_step)

            for q_i in range(self.args.num_Q):
                for p in self.q_net_list[q_i].parameters():
                    p.requires_grad = True

            # target network soft update
            for q_i in range(self.args.num_Q):
                _target_soft_update(self.q_target_net_list[q_i], self.q_net_list[q_i], self.args.tau)

            if self.critic_learning_step % 1000 == 0:
                writer.add_scalar("loss/critic", critic_loss.detach().item(), self.critic_learning_step)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        for i in range(self.args.num_Q):
            _save_model(self.q_net_list[i], self.names[f'q_net_{i}_path'])
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        for i in range(self.args.num_Q):
            _load_model(self.q_net_list[i], self.names[f'q_net_{i}_path'])
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)

    def _value_update(self, state, next_state, action, reward, mask):
        target, sample_idxs = self._redq_q_target(next_state, reward, mask)

        q_prediction_list = []
        for q_i in range(self.args.num_Q):
            q_prediction = self.q_net_list[q_i](state, action)
            q_prediction_list.append(q_prediction)
        q_prediction_cat = T.cat(q_prediction_list, dim=1)
        target = target.expand((-1, self.args.num_Q)) if target.shape[1] == 1 else target

        q_loss_all = self.criterion(q_prediction_cat, target) * self.args.num_Q

        return q_loss_all

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)

        actor_new_q_list = []
        for sample_idx in range(self.args.num_Q):
            new_q = self.q_net_list[sample_idx](state, new_action)
            actor_new_q_list.append(new_q)
        actor_new_q_cat = T.cat(actor_new_q_list, 1)

        ave_q = T.mean(actor_new_q_cat, dim=1, keepdim=True)

        actor_loss = (self.alpha * new_log_prob - ave_q).mean()

        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss

    def _redq_q_target(self, next_state, reward, mask):
        num_mins_to_use = get_probabilistic_num_min(self.args.num_min)
        sample_idxs = np.random.choice(self.args.num_Q, num_mins_to_use, replace=False)
        with T.no_grad():
            if self.args.q_target_mode == 'min':
                """Q target is min of a subset of Q values"""
                next_action, next_log_prob = self.actor(next_state)
                q_prediction_next_list = []
                for sample_idx in sample_idxs:
                    q_prediction_next = self.q_target_net_list[sample_idx](next_state, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = T.cat(q_prediction_next_list, 1)
                min_q, min_indices = T.min(q_prediction_next_cat, dim=1, keepdim=True)
                target = reward + (min_q - self.alpha * next_log_prob) * mask

            if self.args.q_target_mode == 'ave':
                """Q target is average of all Q values"""
                next_action, next_log_prob = self.actor(next_state)
                q_prediction_next_list = []
                for q_i in range(self.args.num_Q):
                    q_prediction_next = self.q_target_net_list[q_i](next_state, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_ave = T.cat(q_prediction_next_list, 1).mean(dim=1).reshape(-1, 1)
                target = reward + (q_prediction_next_ave - self.alpha * next_log_prob) * mask

            if self.args.q_target_mode == 'rem':
                """Q target is random ensemble mixture of Q values"""
                next_action, next_log_prob = self.actor(next_state)
                q_prediction_next_list = []
                for q_i in range(self.args.num_Q):
                    q_prediction_next = self.q_target_net_list[q_i](next_state, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                # apply rem here
                q_prediction_next_cat = T.cat(q_prediction_next_list, 1)
                rem_weight = T.Tensor(np.random.uniform(0, 1, q_prediction_next_cat.shape)).to(device=self.args.device)
                normalize_sum = rem_weight.sum(1).reshape(-1, 1).expand(-1, self.args.num_Q)
                rem_weight = rem_weight / normalize_sum
                q_prediction_next_rem = (q_prediction_next_cat * rem_weight).sum(dim=1).reshape(-1, 1)
                target = reward + (q_prediction_next_rem - self.alpha * next_log_prob) * mask

        return target, sample_idxs

