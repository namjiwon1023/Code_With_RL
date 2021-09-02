import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import copy
import gym
import dmc2gym

from network import Actor, Critic, Encoder, RandomShiftsAug
from replaybuffer import ReplayBuffer
import utils
import dmc

class DrQAgent:
    def __init__(self, args):
        self.args = args
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)
        self.encoder_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_encoder)

        utils.set_seed_everywhere(args.seed)
        # Environment setting
        self.env = dmc.make(self.args.env_name,
                            self.args.frame_stack,
                            self.args.action_repeat,
                            self.args.seed)

        self.n_states = self.env.observation_spec.shape
        self.n_actions = self.env.action_spec.shape

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args.buffer_size, self.device)
        self.transition = list()

        # actor-critic net setting
        self.encoder = Encoder(self.n_states)
        self.actor = Actor(self.encoder.repr_dim, self.n_actions, self.device, self.args.feature_dim, self.args.hidden_dim)
        self.critic = Critic(self.encoder.repr_dim, self.n_actions, self.device, self.args.feature_dim, self.args.hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.aug = RandomShiftsAug(pad=self.args.image_pad)

        # loss function
        self.criterion = nn.MSELoss()

        # Temperature Coefficient
        self.log_alpha = T.tensor(np.log(self.args.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -self.n_actions[0]
        self.alpha = self.log_alpha.exp()

        # optimizer setting
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.args.encoder_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        self.total_step = 0
        self.learning_step = 0

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)


    def choose_action(self, state, evaluate=False):
        if self.total_step <= self.args.init_random_steps and not evaluate:
            choose_action = self.env.action_space.sample()
        else :
            if evaluate:
                obs = self.encoder(T.as_tensor((state,), dtype=T.float32, device=self.device))
                choose_action, _ = self.actor(obs, evaluate=True, with_logprob=False)
                choose_action = choose_action[0].detach().cpu().numpy()
            else:
                obs = self.encoder(T.as_tensor((state,), dtype=T.float32, device=self.device))
                choose_action, _ = self.actor(obs, evaluate=False, with_logprob=True)
                choose_action = choose_action[0].detach().cpu().numpy()
        if not evaluate:
            self.transition = [state, choose_action]
        return choose_action

    def learn(self, writer):
        if self.total_step % self.args.update_frequency == 0:
            self.learning_step += 1
            # TD error
            # update value
            state, action, reward, next_state, mask = self.memory.sample_batch(self.args.batch_size)

            # augment
            state = self.aug(state.float())
            next_state = self.aug(next_state.float())

            # encode
            state = self.encoder(state)
            with T.no_grad():
                next_state = self.encode(next_state)

            critic_loss = self._value_update(state, action, reward, next_state, mask)

            self.encoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.encoder_optimizer.step()

            actor_loss, new_log_prob = self._policy_update(state.detach())

            # update Policy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update Temperature Coefficient
            alpha_loss = self._temperature_update(new_log_prob)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

            # target network soft update
            self._target_soft_update(self.critic_target, self.critic, self.args.tau)

            if self.learning_step % 1000 == 0:
                writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)
                writer.add_scalar("loss/actor", actor_loss.item(), self.learning_step)
                writer.add_scalar("loss/alpha", alpha_loss.item(), self.learning_step)

    def save_models(self):
        print('------ Save model ------')
        utils._save_model(self.actor, self.actor_path)
        utils._save_model(self.critic, self.critic_path)
        utils._save_model(self.encoder, self.encoder_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load model ------')
        utils._load_model(self.actor, self.actor_path)
        utils._load_model(self.critic, self.critic_path)
        utils._load_model(self.encoder, self.encoder_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)

    # target network soft update
    def _target_soft_update(self, target, net, tau):
        for t_p, l_p in zip(target.parameters(), net.parameters()):
            t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def _value_update(self, state, action, reward, next_state, mask):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            state, action, reward, next_state, mask = buffer.sample_batch(batch_size)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = self.criterion(current_q1, target_q) + self.criterion(current_q2, target_q)

        return critic_loss

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state, detach_encoder=True)
        q_1, q_2 = self.critic(state, new_action, detach_encoder=True)
        q = T.min(q_1, q_2)
        # update actor network
        actor_loss = (self.alpha * new_log_prob - q).mean()
        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss
