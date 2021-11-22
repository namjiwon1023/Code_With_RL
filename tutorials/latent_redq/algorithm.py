# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import copy

import numpy as np
import torch as T
from torch.optim import Adam
import torch.nn as nn

from replaybuffer import ReplayBuffer
from network.sac import GaussianPolicy, QNetwork
from network.latent import LatentModel
from utils import create_input, make_dmc, eval_mode, _random_seed


class latentREDQAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.env_name, 'REDQ_actor.pth')
        self.latent_path = os.path.join(args.save_dir +'/' + args.env_name, 'latent.pth')
        self.encoder_path = os.path.join(args.save_dir +'/' + args.env_name, 'encoder.pth')

        self.env = make_dmc(domain_name=args.env_name,
                            task_name=args.task_name,
                            action_repeat=args.action_repeat,
                            image_size=64,
                            )
        self.env_test = make_dmc(
                            domain_name=args.env_name,
                            task_name=args.task_name,
                            action_repeat=args.action_repeat,
                            image_size=64,
                            )

        _random_seed(self.env, self.env_test, self.args.seed)

        self.critic_learning_step, self.actor_learning_step = 0, 0
        self.learning_steps_latent = 0
        self.total_step = 0

        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape

        self.action_repeat = self.args.action_repeat

        self.initial_collection_steps = args.initial_collection_steps
        self.delay_update_steps = args.initial_collection_steps if args.delay_update_steps == 'auto' else args.delay_update_steps

        self.device = args.device
        self.gamma = args.gamma
        self.batch_size_sac = args.batch_size_sac
        self.batch_size_latent = args.batch_size_latent
        self.num_sequences = args.num_sequences
        self.tau = args.tau

        # Replay buffer.
        self.buffer = ReplayBuffer(self.args.buffer_size, self.num_sequences, self.state_shape, self.action_shape, self.device)

        # Networks.
        self.actor = GaussianPolicy(self.action_shape, self.num_sequences, args.feature_dim, self.device, args.hidden_units)

        self.names = locals()
        for i in range(self.args.num_Q):
            self.names[f'q_net_{i}_path'] = os.path.join(args.save_dir +'/' + args.env_name, args.file_critic + '_%d' % i)

        # Collection of Q Network
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(self.args.num_Q):
            new_q_net = QNetwork(self.action_shape, args.z1_dim, args.z2_dim, self.device, args.hidden_units)
            self.q_net_list.append(new_q_net)
            # new_q_target_net = copy.deepcopy(new_q_net)
            new_q_target_net = QNetwork(self.action_shape, args.z1_dim, args.z2_dim, self.device, args.hidden_units)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            # disable_gradients(new_q_target_net)
            self.q_target_net_list.append(new_q_target_net)

        self.latent = LatentModel(self.state_shape, self.action_shape, self.device, args.feature_dim, args.z1_dim, args.z2_dim, args.hidden_units)

        # Target entropy is -|A|.
        # Temperature Coefficient
        if args.target_entropy == 'auto':
            self.target_entropy = -float(self.action_shape[0])

        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = T.zeros(1, requires_grad=True, device=args.device)

        # Optimizers.
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr)
        self.q_optimizer_list = []
        for q_i in range(self.args.num_Q):
            self.q_optimizer_list.append(Adam(self.q_net_list[q_i].parameters(), lr=args.critic_lr))
        self.alpha_optimizer = Adam([self.log_alpha], lr=args.alpha_lr)
        self.latent_optimizer = Adam(self.latent.parameters(), lr=args.latent_lr)

        self.criterion = nn.MSELoss()

        self.train()
        for i in range(self.args.num_Q):
            self.q_target_net_list[i].train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.latent.train(training)
        for i in range(self.args.num_Q):
            self.q_net_list[i].train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def preprocess(self, ob):
        state = T.tensor(ob.state, dtype=T.uint8, device=self.device).float().div_(255.0)
        with T.no_grad():
            feature = self.latent.encoder(state).view(1, -1)
        action = T.tensor(ob.action, dtype=T.float, device=self.device)
        feature_action = T.cat([feature, action], dim=1)
        return feature_action

    def select_exploration_action(self, ob):
        with T.no_grad():
            if self.total_step <= self.initial_collection_steps:
                select_exploration_action = self.env.action_space.sample()
            else:
                feature_action = self.preprocess(ob)
                select_exploration_action = self.actor.sample(feature_action)[0]
                select_exploration_action = select_exploration_action.cpu().numpy()[0]
        return select_exploration_action

    def select_test_action(self, ob):
        with T.no_grad():
            feature_action = self.preprocess(ob)
            select_test_action = self.actor(feature_action)
            select_test_action = select_test_action.cpu().numpy()[0]
        return select_test_action


    def step(self, agent, ob, t):
        t += 1
        self.total_step += 1

        with eval_mode(agent):
            action = agent.select_exploration_action(ob)

        state, reward, done, _ = self.env.step(action)
        mask = False if t == self.env._max_episode_steps else done
        ob.append(state, action)
        self.buffer.append(action, reward, mask, state, done)

        if done:
            t = 0
            state = self.env.reset()
            ob.reset_episode(state)
            self.buffer.reset_episode(state)

        return t

    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state, action, reward, done = self.buffer.sample_latent(self.batch_size_latent)
        loss_kld, loss_image, loss_reward = self.latent.calculate_loss(state, action, reward, done)

        self.latent_optimizer.zero_grad()
        (loss_kld + loss_image + loss_reward).backward()
        self.latent_optimizer.step()

        if self.learning_steps_latent % 1000 == 0:
            writer.add_scalar("loss/kld", loss_kld.item(), self.learning_steps_latent)
            writer.add_scalar("loss/reward", loss_reward.item(), self.learning_steps_latent)
            writer.add_scalar("loss/image", loss_image.item(), self.learning_steps_latent)

    def prepare_batch(self, state, action):
        with T.no_grad():
            # f(1:t+1)
            feature = self.latent.encoder(state)
            # z(1:t+1)
            Z = T.cat(self.latent.sample_posterior(feature, action)[2:4], dim=-1)

        # z(t), z(t+1)
        z_t, next_z = Z[:, -2], Z[:, -1]
        # a(t)
        a_t = action[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = create_input(feature, action)

        return z_t, next_z, a_t, feature_action, next_feature_action

    def update_sac(self, writer):
        num_update = 0 if self.total_step <= self.delay_update_steps else self.args.utd_ratio
        for i_update in range(num_update):
            self.critic_learning_step += 1
            state, action, reward, done = self.buffer.sample_sac(self.batch_size_sac)
            z_t, next_z, a_t, feature_action, next_feature_action = self.prepare_batch(state, action)
            # TD error
            # update value
            critic_loss = self._value_update(z_t, next_z, a_t, next_feature_action, reward, done)

            for q_i in range(self.args.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            critic_loss.backward()

            if ((i_update + 1) % self.args.policy_update_delay == 0) or i_update == num_update - 1:
                self.actor_learning_step += 1
                actor_loss, new_log_prob = self._policy_update(z_t, feature_action)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                for sample_idx in range(self.args.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)

                # update Temperature Coefficient
                alpha_loss = self._temperature_update(new_log_prob)

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                if self.actor_learning_step % 1000 == 0:
                    writer.add_scalar("loss/actor", actor_loss.detach().item(), self.actor_learning_step)
                    writer.add_scalar("loss/alpha", alpha_loss.detach().item(), self.actor_learning_step)
                    writer.add_scalar("value/alpha", self.alpha.detach().item(), self.actor_learning_step)


            for q_i in range(self.args.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.args.policy_update_delay == 0) or i_update == num_update - 1:
                self.actor_optimizer.step()

            for q_i in range(self.args.num_Q):
                self._target_soft_update(self.q_target_net_list[q_i], self.q_net_list[q_i], self.args.tau)

            if self.critic_learning_step % 1000 == 0:
                writer.add_scalar("loss/critic", critic_loss.detach().item(), self.critic_learning_step)

    def _value_update(self, z_t, next_z, a_t, next_feature_action, reward, done):

        target, sample_idxs = self._redq_q_target(next_feature_action, next_z, reward, done)

        q_prediction_list = []
        for q_i in range(self.args.num_Q):
            q_prediction = self.q_net_list[q_i](z_t, a_t)
            q_prediction_list.append(q_prediction)
        q_prediction_cat = T.cat(q_prediction_list, dim=1)
        target = target.expand((-1, self.args.num_Q)) if target.shape[1] == 1 else target

        q_loss_all = self.criterion(q_prediction_cat, target) * self.args.num_Q

        return q_loss_all

    def _policy_update(self, z_t, feature_action):

        new_action, new_log_prob = self.actor.sample(feature_action)
        actor_new_q_list = []
        for sample_idx in range(self.args.num_Q):
            self.q_net_list[sample_idx].requires_grad_(False)
            new_q = self.q_net_list[sample_idx](z_t, new_action)
            actor_new_q_list.append(new_q)
        actor_new_q_cat = T.cat(actor_new_q_list, dim=1)

        ave_q = T.mean(actor_new_q_cat, dim=1, keepdim=True)

        actor_loss = (self.alpha * new_log_prob - ave_q).mean()

        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss

    def _redq_q_target(self, next_feature_action, next_z, reward, done):
        num_mins_to_use = self._get_probabilistic_num_min(self.args.num_min)
        sample_idxs = np.random.choice(self.args.num_Q, num_mins_to_use, replace=False)
        with T.no_grad():
            if self.args.q_target_mode == 'min':
                """Q target is min of a subset of Q values"""
                next_action, next_log_prob = self.actor.sample(next_feature_action)
                q_prediction_next_list = []
                for sample_idx in sample_idxs:
                    q_prediction_next = self.q_target_net_list[sample_idx](next_z, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = T.cat(q_prediction_next_list, dim=1)
                min_q, min_indices = T.min(q_prediction_next_cat, dim=1, keepdim=True)
                target = reward + (min_q - self.alpha * next_log_prob) * self.gamma * (1.0 - done)

            if self.args.q_target_mode == 'ave':
                """Q target is average of all Q values"""
                next_action, next_log_prob = self.actor.sample(next_feature_action)
                q_prediction_next_list = []
                for q_i in range(self.args.num_Q):
                    q_prediction_next = self.q_target_net_list[q_i](next_z, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_ave = T.cat(q_prediction_next_list, dim=1).mean(dim=1).reshape(-1, 1)
                target = reward + (q_prediction_next_ave - self.alpha * next_log_prob) * self.gamma * (1.0 - done)

            if self.args.q_target_mode == 'rem':
                """Q target is random ensemble mixture of Q values"""
                next_action, next_log_prob = self.actor.sample(next_feature_action)
                q_prediction_next_list = []
                for q_i in range(self.args.num_Q):
                    q_prediction_next = self.q_target_net_list[q_i](next_z, next_action)
                    q_prediction_next_list.append(q_prediction_next)
                # apply rem here
                q_prediction_next_cat = T.cat(q_prediction_next_list, dim=1)
                rem_weight = T.Tensor(np.random.uniform(0, 1, q_prediction_next_cat.shape)).to(device=self.args.device)
                normalize_sum = rem_weight.sum(1).reshape(-1, 1).expand(-1, self.args.num_Q)
                rem_weight = rem_weight / normalize_sum
                q_prediction_next_rem = (q_prediction_next_cat * rem_weight).sum(dim=1).reshape(-1, 1)
                target = reward + (q_prediction_next_rem - self.alpha * next_log_prob) * self.gamma * (1.0 - done)

        return target, sample_idxs

    def _get_probabilistic_num_min(self, num_mins):
        # allows the number of min to be a float
        floored_num_mins = np.floor(num_mins)
        if num_mins - floored_num_mins > 0.001:
            prob_for_higher_value = num_mins - floored_num_mins
            if np.random.uniform(0, 1) < prob_for_higher_value:
                return int(floored_num_mins+1)
            else:
                return int(floored_num_mins)
        else:
            return num_mins

    def _target_soft_update(self, target_net, eval_net, tau):
        for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
            t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def save_models(self):
        print('------ Save models ------')
        T.save(self.latent.encoder.state_dict(), self.encoder_path)
        T.save(self.latent.state_dict(), self.latent_path)
        T.save(self.actor.state_dict(), self.actor_path)
        for i in range(self.args.num_Q):
            self._save_model(self.q_net_list[i], self.names[f'q_net_{i}_path'])
        checkpoint = os.path.join(self.args.save_dir +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load models ------')
        self.actor.load_state_dict(T.load(self.actor_path))
        self.latent.load_state_dict(T.load(self.latent_path))
        self.latent.encoder.load_state_dict(T.load(self.encoder_path))
        for i in range(self.args.num_Q):
            self._load_model(self.q_net_list[i], self.names[f'q_net_{i}_path'])
        checkpoint = os.path.join(self.args.save_dir +'/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)

    def _save_model(self, net, dirpath):
        T.save(net.state_dict(), dirpath)

    def _load_model(self, net, dirpath):
        net.load_state_dict(T.load(dirpath))