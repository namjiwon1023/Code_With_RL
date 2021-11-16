# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import math

import numpy as np
import torch as T
import torch.nn as nn
from torch.nn import functional as F

from utils import build_mlp, calculate_kl_divergence, initialize_weight


class FixedGaussian(nn.Module):
    """
    Fixed diagonal gaussian distribution.
    固定 对角 高斯分布
    latent z1 initialize network
    z1 ~ p(z1)
    output : z1_mean : [0], z1_std : [1]
    z1\1 ~ P(z1\1) 先验
    """

    def __init__(self, output_dim, std):
        super(FixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = T.zeros(x.size(0), self.output_dim, device=x.device)
        std = T.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class Gaussian(nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    具有 状态相关方差 的 对角高斯分布
    output : mean, std
    activation: LeakyReLU
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Gaussian, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(0.2),
            ).apply(initialize_weight)

    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        mean, std = T.chunk(x, 2, dim=-1)    # or network[output_dim : 1]layer * 2
        std = F.softplus(std) + 1e-5    # in the paper std output activation:  softplus
        return mean, std

class Decoder(nn.Module):
    """
    Decoder.
    解码器
    P(x|z)
    input_dim = z1:32 + z2:256 = total_Z_dim:288
    output_dim = RGB_Channel_Dim:3
    in the paper: Except for the first layer， Add stride to 2
    """

    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)
        self.std = std


    def forward(self, x):
        B, S, latent_dim = x.size()    # Batch_size, Sequence, latent_dim
        x = x.view(B * S, latent_dim, 1, 1)   # x -> (N, C, H, W)
        x = self.net(x)       # x = (B*S, 3, 64, 64) | network input size:(N, C, H, W)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H) # (B, S, 3, 64, 64)
        return x, T.ones_like(x).mul_(self.std)      # output : image, std


class Encoder(nn.Module):
    """
    Encoder.
    编码器
    q(z1\1 | x1\1) : 后验模型
    (3, 64, 64) - > (256, 1, 1)
    """

    def __init__(self, input_dim=3, output_dim=256):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)


    def forward(self, x):
        B, S, C, H, W = x.size()      # Batch_size, Sequence, Channel, Height, Width
        x = x.view(B * S, C, H, W)    # -> (N, C, H, W)
        x = self.net(x)
        x = x.view(B, S, -1)            # (B, S, 256)
        return x    # output: feature X


class LatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics and the reward.
    用于估计 潜在动态 和 奖励 的 随机潜在变量模型
    parameters setting in paper :
    feature_dims : 256
    z1_dim : 32
    z2_dim : 256
    network hidden units : 256
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        device,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
    ):
        super(LatentModel, self).__init__()

        ''' Generative Model '''
        ''' Prior probability model '''

        # p(z1(0)) = N(0, I) | p(z1\1) = N(0, I)
        # 先验函数， Z1初始分布由多元正态分布 N ~(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0)) | p(z2\1|z1\1)
        # z2的初始分布由z1决定
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        # 从概率图可知由latent z2和 action生成下一时间步的z1
        self.z1_prior = Gaussian(z2_dim + action_shape[0], z1_dim, hidden_units)
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        # 从概率图可知由latent z2和 action和下一时间步的z1 生成下一时间步的z2
        self.z2_prior = Gaussian(z1_dim + z2_dim + action_shape[0], z2_dim, hidden_units)
        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = Gaussian(2 * z1_dim + 2 * z2_dim + action_shape[0], 1, hidden_units)
        # p(x(t) | z1(t), z2(t))
        # 方差是0.1 , 偏差是方差的平方根 所以用np.sqrt()
        # #选择范围DM_control(0.04, 0.1, 0.4) | OpenAI_gym(0.1)
        self.decoder = Decoder(z1_dim + z2_dim, state_shape[0], std=np.sqrt(0.1))

        ''' Posterior probability model '''
        ''' In the paper, the initial prior model and posterior model of z2 are the same '''
        # q(z1(0) | feat(0)), feature是经过Encoder处理过得 x
        # z1(0) ~ q(z1(0)|x(0))
        self.z1_posterior_init = Gaussian(feature_dim, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        # z2的初始化先验后验是相同的
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(feature_dim + z2_dim + action_shape[0], z1_dim, hidden_units)
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        # z2的初始化先验后验是相同的
        self.z2_posterior = self.z2_prior

        # feat(t) = Encoder(x(t))
        # 将图片进行编码生成 X
        self.encoder = Encoder(state_shape[0], feature_dim)

        self.apply(initialize_weight)
        self.device = device
        self.to(self.device)


    def sample_prior(self, actions, z2):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions[:, 0]) # size : [Batch_size, action_shape:6] 所有的最初动作
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean, z1_std = self.z1_prior(T.cat([z2[:, : actions.size(1)], actions], dim=-1))
        # Concatenate initial and consecutive latent variables
        z1_mean = T.cat([z1_mean_init.unsqueeze(1), z1_mean], dim=1)
        z1_std = T.cat([z1_std_init.unsqueeze(1), z1_std], dim=1)

        return (z1_mean, z1_std)


    def sample_posterior(self, features, actions):
        ''' reparameterization trick '''
        ''' value = mu + std * noise | noise ~ N(0,1) '''
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features[:, 0]) # size : [Batch_size, features_dim]
        z1 = z1_mean + T.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + T.randn_like(z2_std) * z2_std

        z1_mean_list = [z1_mean]
        z1_std_list = [z1_std]
        z1_list = [z1]
        z2_list = [z2]

        for t in range(1, actions.size(1) + 1):     # state sequence length 9(Because have state(t + 1)), action sequence length 8(just t)
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(T.cat([features[:, t], z2, actions[:, t - 1]], dim=1))
            z1 = z1_mean + T.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(T.cat([z1, z2, actions[:, t - 1]], dim=1))
            z2 = z2_mean + T.randn_like(z2_std) * z2_std

            z1_mean_list.append(z1_mean)
            z1_std_list.append(z1_std)
            z1_list.append(z1)
            z2_list.append(z2)

        z1_mean_list = T.stack(z1_mean_list, dim=1)
        z1_std_list = T.stack(z1_std_list, dim=1)
        z1_list = T.stack(z1_list, dim=1)
        z2_list = T.stack(z2_list, dim=1)

        return (z1_mean_list, z1_std_list, z1_list, z2_list)


    def calculate_loss(self, state, action, reward, done):
        # Calculate the sequence of features.
        # feat(t) = Encoder(x(t))
        feature = self.encoder(state)

        # Sample from latent variable model.
        z1_mean_post, z1_std_post, z1, z2 = self.sample_posterior(feature, action)
        z1_mean_pri, z1_std_pri = self.sample_prior(action, z2)

        # Calculate KL divergence loss.
        # in the paper DKL(log(q(z(t+1)))||log(p(z(t+1))))
        loss_kld = calculate_kl_divergence(z1_mean_post, z1_std_post, z1_mean_pri, z1_std_pri).mean(dim=0).sum()

        # Prediction loss of images.
        Z = T.cat([z1, z2], dim=-1)
        state_mean, state_std = self.decoder(Z)
        # Z-score normalization
        state_noise = (state - state_mean) / (state_std + 1e-8)
        # log likelihood
        log_likelihood = (-0.5 * state_noise.pow(2) - state_std.log()) - 0.5 * math.log(2 * math.pi)
        loss_image = -log_likelihood.mean(dim=0).sum()

        # Prediction loss of rewards.
        # z(0:t), action(0:t), z(1,t+1)
        x = T.cat([Z[:, :-1], action, Z[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean, reward_std = self.reward(x.view(B * S, X)) # network input size (N,x)
        reward_mean = reward_mean.view(B, S, 1)
        reward_std = reward_std.view(B, S, 1)
        # Z-score normalization
        reward_noise = (reward - reward_mean) / (reward_std + 1e-8)
        log_likelihood_reward = (-0.5 * reward_noise.pow(2) - reward_std.log()) - 0.5 * math.log(2 * math.pi)
        loss_reward = -log_likelihood_reward.mul_(1 - done).mean(dim=0).sum()
        return loss_kld, loss_image, loss_reward
