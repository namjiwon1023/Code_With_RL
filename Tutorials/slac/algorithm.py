import os
import copy

import numpy as np
import torch as T
from torch.optim import Adam
import torch.nn as nn

from slac.replaybuffer import ReplayBuffer
from slac.network.sac import GaussianPolicy, TwinnedQNetwork
from slac.network.latent import LatentModel
from slac.utils import create_input, make_dmc


class SlacAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.env_name, 'sac_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.env_name, 'sac_critic.pth')
        self.latent_path = os.path.join(args.save_dir +'/' + args.env_name, 'latent.pth')
        self.encoder_path = os.path.join(args.save_dir +'/' + args.env_name, 'encoder.pth')

        np.random.seed(args.seed)
        T.manual_seed(args.seed)
        T.cuda.manual_seed(args.seed)
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

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.total_step = 0

        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape

        self.action_repeat = self.args.action_repeat

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
        self.critic = TwinnedQNetwork(self.action_shape, args.z1_dim, args.z2_dim, self.device, args.hidden_units)

        self.latent = LatentModel(self.state_shape, self.action_shape, self.device, args.feature_dim, args.z1_dim, args.z2_dim, args.hidden_units)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Target entropy is -|A|.
        self.target_entropy = -float(self.action_shape[0])
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = T.zeros(1, requires_grad=True, device=args.device)
        with T.no_grad():
            self.alpha = self.log_alpha.exp()

        # Optimizers.
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.critic_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=args.alpha_lr)
        self.latent_optimizer = Adam(self.latent.parameters(), lr=args.latent_lr)

        self.criterion = nn.MSELoss()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if os.path.exists(self.model_path + '/sac_actor.pth'):
            self.load_models()

    def preprocess(self, ob):
        state = T.tensor(ob.state, dtype=T.uint8, device=self.device).float().div_(255.0)
        with T.no_grad():
            feature = self.latent.encoder(state).view(1, -1)
        action = T.tensor(ob.action, dtype=T.float, device=self.device)
        feature_action = T.cat([feature, action], dim=1)
        return feature_action

    def choose_action(self, ob, evaluate=False):
        if self.total_step <= self.args.initial_collection_steps and not evaluate:
            choose_action = self.env.action_space.sample()
        else:
            feature_action = self.preprocess(ob)
            with T.no_grad():
                if evaluate:
                    choose_action = self.actor(feature_action)
                    choose_action = choose_action.cpu().numpy()[0]
                else:
                    choose_action = self.actor.sample(feature_action)[0]
                    choose_action = choose_action.cpu().numpy()[0]
        return choose_action

    def step(self, ob, t):
        t += 1
        self.total_step += 1

        action = self.choose_action(ob, evaluate=False)

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

    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state, action, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        z_t, next_z, a_t, feature_action, next_feature_action = self.prepare_batch(state, action)

        self.update_critic(z_t, next_z, a_t, next_feature_action, reward, done, writer)
        self.update_actor(z_t, feature_action, writer)
        self._target_soft_update(self.critic_target, self.critic, self.tau)

    def prepare_batch(self, state, action):
        with T.no_grad():
            # f(1:t+1)
            feature = self.latent.encoder(state)
            # z(1:t+1)
            Z = T.cat(self.latent.sample_posterior(feature, action)[2:], dim=-1)

        # z(t), z(t+1)
        z_t, next_z = Z[:, -2], Z[:, -1]
        # a(t)
        a_t = action[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature, action)

        return z_t, next_z, a_t, feature_action, next_feature_action

    def update_critic(self, z_t, next_z, a_t, next_feature_action, reward, done, writer):
        with T.no_grad():
            next_action, next_log_prob = self.actor.sample(next_feature_action)
            next_target_q1, next_target_q2 = self.critic_target(next_z, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            target_q = reward + self.gamma * (next_target_q - self.alpha * next_log_prob) * (1.0 - done)
        current_q1, current_q2 = self.critic(z_t, a_t)
        loss_critic = self.criterion(current_q1, target_q) + self.criterion(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        if self.learning_steps_sac % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic.item(), self.learning_steps_sac)

    def update_actor(self, z_t, feature_action, writer):
        new_action, new_log_prob = self.actor.sample(feature_action)
        q_1, q_2 = self.critic(z_t, new_action)
        q = T.min(q_1, q_2)
        loss_actor = (self.alpha * new_log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            entropy = -new_log_prob.detach().mean()

        loss_alpha = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()

        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        if self.learning_steps_sac % 1000 == 0:
            writer.add_scalar("loss/actor", loss_actor.item(), self.learning_steps_sac)
            writer.add_scalar("loss/alpha", loss_alpha.item(), self.learning_steps_sac)
            writer.add_scalar("stats/alpha", self.alpha.item(), self.learning_steps_sac)
            writer.add_scalar("stats/entropy", entropy.item(), self.learning_steps_sac)

    def save_models(self):
        print('------ Save models ------')
        T.save(self.latent.encoder.state_dict(), self.encoder_path)
        T.save(self.latent.state_dict(), self.latent_path)
        T.save(self.actor.state_dict(), self.actor_path)
        T.save(self.critic.state_dict(), self.critic_path)

    def load_models(self):
        print('------ load models ------')
        self.actor.load_state_dict(T.load(self.actor_path))
        self.critic.load_state_dict(T.load(self.critic_path))
        self.latent.load_state_dict(T.load(self.latent_path))
        self.latent.encoder.load_state_dict(T.load(self.encoder_path))

    def _target_soft_update(self, target_net, eval_net , tau=None):
        if tau == None:
            tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)
