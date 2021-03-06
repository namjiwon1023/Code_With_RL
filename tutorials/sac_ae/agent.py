import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os
import copy
from network.AutoEncoder import Decoder
from network.AutoEncoder import Encoder
from network.sac_ae import ActorSAC, Critic
import dmc2gym
import math
from utils.utils import ReplayBuffer, weight_init, update_params
import utils.utils as utils

class SacAeAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-4,
        alpha_beta=0.5,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.01,
        critic_target_update_freq=2,
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=1e-6,
        decoder_weight_lambda=1e-7,
    ):
        self.device = device
        self.discount = discount

        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau

        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq

        self.decoder_latent_lambda = decoder_latent_lambda

        self.actor = ActorSAC(obs_shape, action_shape, encoder_feature_dim, device)
        self.critic = Critic(obs_shape, action_shape, encoder_feature_dim, device)

        self.critic_target = copy.deepcopy(self.critic)

        # tie encoders between actor and critic
        self.actor.encoder.sharing_parameters_actor_critic_encoder(self.critic.encoder)

        self.log_alpha = T.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # create decoder
        self.decoder = Decoder(obs_shape, encoder_feature_dim, device)
        self.decoder.apply(weight_init)

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = optim.Adam(self.critic.encoder.parameters(), lr=encoder_lr)

        # optimizer for decoder
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=decoder_lr, weight_decay=decoder_weight_lambda)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))

        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with T.no_grad():
            obs = T.as_tensor(obs, device=self.device)
            obs = obs.unsqueeze(0)
            action, _, _ = self.actor(obs, evaluate=True, with_logprob=False)
            return action.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with T.no_grad():
            obs = T.as_tensor(obs, device=self.device)
            obs = obs.unsqueeze(0)
            action, _, _ = self.actor(obs, evaluate=False, with_logprob=True)
            return action.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        with T.no_grad():
            policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = T.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        update_params(self.critic_optimizer, critic_loss)

    def update_actor_and_alpha(self, obs, step):
        # detach encoder, so we don't update it with the actor loss
        pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = T.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)

        # optimize the actor
        update_params(self.actor_optimizer, actor_loss)

        alpha_loss = (self.alpha *
                    (-log_pi - self.target_entropy).detach()).mean()

        update_params(self.log_alpha_optimizer, alpha_loss)

    def update_decoder(self, obs, target_obs, step):
        h = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)

        # target obs vs decoder obs
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(obs, action, reward, next_obs, not_done, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)

        if step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, step)

    def save(self, model_dir, step):
        T.save(self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step))
        T.save(self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step))
        T.save(self.decoder.state_dict(), '%s/decoder_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(T.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(T.load('%s/critic_%s.pt' % (model_dir, step)))
        self.decoder.load_state_dict(T.load('%s/decoder_%s.pt' % (model_dir, step)))