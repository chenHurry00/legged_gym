# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from .utility.normalizer import EmpiricalNormalization
from .utility.common_modules import MAE, VQVAE, VQVAE_CNN, VQVAE_EMA, VQVAE_RNN, AutoEncoder, BetaVAE, \
    CnnHistoryEncoder, MixedLayerNormMlp, MixedLipMlp, MixedMlp, RnnBarlowTwinsStateHistoryEncoder, \
    RnnDoubleHeadEncoder, RnnEncoder, RnnStateHistoryEncoder, StateHistoryEncoder, VQVAE_Trans, VQVAE_vel, \
    VQVAE_vel_conv, get_activation, mlp_batchnorm_factory, mlp_factory, mlp_layernorm_factory


class LPActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actions,
                        num_proprio,
                        history_length,
                        num_scan,
                        #num_priv, # TODO：add priv
                        history_latent_size=32,
                        scan_latent_size=256,
                        proprio_vel_latent_size=32,
                        #priv_latent_size=32,
                        history_hidden_dims=[256, 64],
                        actor_hidden_dims=[256, 128, 64],
                        scan_hidden_dims=[256,256],
                        proprio_vel_hidden_dims=[128,64],
                        #privileged_hidden_dims=[128, 64],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(LPActorCritic, self).__init__()

        self.num_prop = num_proprio
        self.history_length = history_length - 1
        self.num_scan = num_scan
        # self.num_priv = num_priv

        activation = get_activation(activation)

        self.history_layer = build_mlp(self.history_length * num_proprio, history_hidden_dims, history_latent_size,
                                       activation)
        self.scan_layer = build_mlp(num_scan, scan_hidden_dims, scan_latent_size, activation)
        self.proprio_vel_layer = build_mlp(num_proprio + 3, proprio_vel_hidden_dims, proprio_vel_latent_size,
                                           activation)
        self.vel_layer = nn.Linear(history_latent_size, 3)
        # self.pivileged_layer = build_mlp(num_priv,privileged_hidden_dims,priv_latent_size,activation)

        # history encoder
        self.latent_layer = nn.Sequential(nn.Linear(history_latent_size, 32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32, history_latent_size))
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                                              input_dims=history_latent_size,
                                                              out_dims=64,
                                                              hidden_dims=[64],
                                                              bias=False))
        self.bn = nn.BatchNorm1d(64, affine=False)

        # Policy
        self.actor = build_mlp(self.num_prop + 3 + history_latent_size, actor_hidden_dims, num_actions, activation)

        # Value function
        self.critic = build_mlp(scan_latent_size + proprio_vel_latent_size, critic_hidden_dims, 1, activation)

        networks = {
            "History": self.history_layer,
            "Latent_z": self.latent_layer,
            "Actor": self.actor,

            "Scan": self.scan_layer,
            "Proprio Vel": self.proprio_vel_layer,
            # "Privileged": self.privileged_layer,
            "Critic": self.critic
        }
        for name, net in networks.items():
            print(f"{name} MLP: {net}")

        # normalizer
        self.obs_prop = None
        self.obs_hist = None
        self.obs_normalizer = EmpiricalNormalization(shape=num_proprio)
        self.obs_vel_normalizer = EmpiricalNormalization(shape=num_proprio+3)
        self.scan_normalizer = EmpiricalNormalization(shape=num_scan)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.obs_hist = observations[:, :self.num_prop * self.history_length]  # obs_hist size = history_length-1
        obs_prop = observations[:, :self.num_prop]
        batch_size = obs_prop.shape[0]

        obs_prop_norm, obs_hist_norm = self.normalize(obs_prop, self.obs_hist)

        obs_hist_flat = obs_hist_norm.reshape(batch_size, -1)  # (B, L*P)
        with torch.no_grad():
            latent = self.history_layer(obs_hist_flat)
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
        actor_input = torch.cat([obs_prop_norm, vel, z], dim=1)
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        self.obs_hist = observations[:, :self.num_prop * self.history_length]  # obs_hist size = history_length-1
        obs_prop = observations[:, :self.num_prop]
        batch_size = obs_prop.shape[0]

        obs_prop_norm, obs_hist_norm = self.normalize(obs_prop, self.obs_hist)

        obs_hist_flat = obs_hist_norm.reshape(batch_size, -1)  # (B, L*P)
        with torch.no_grad():
            latent = self.history_layer(obs_hist_flat)
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
        actor_input = torch.cat([obs_prop_norm, vel, z], dim=1)

        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs_prop_vel = critic_observations[:, :self.num_prop + 3]
        obs_scan = critic_observations[:, self.num_prop + 3:self.num_prop + self.num_scan + 3]
        # obs_priv = critic_observations[:, self.num_prop+self.num_scan+3:self.num_prop+self.num_scan+self.num_priv+3]

        # Normalize inputs
        obs_prop_vel_norm = torch.cat(
            [self.obs_normalizer(obs_prop_vel[:, :self.num_prop]), obs_prop_vel[:, self.num_prop:]],
            dim=1)  # Normalize proprio, keep vel as is
        obs_scan_norm = self.scan_normalizer(obs_scan)

        prop_vel_latent = self.proprio_vel_layer(obs_prop_vel_norm)
        scan_latent = self.scan_layer(obs_scan_norm)
        # priv_latent = self.privileged_layer(obs_priv)
        value_input = torch.cat([prop_vel_latent, scan_latent], dim=1)

        value = self.critic(value_input)
        return value

    def imitation_learning_loss(self, obs, critic_obs):
        obs_prop = obs[:, :self.num_prop]
        obs_old_his = obs[:, self.num_prop:self.num_prop * (self.history_length + 1)]
        obs_prop_vel = critic_obs[:, :self.num_prop + 3]
        loss = self.BarlowTwinsLoss(obs_prop, obs_old_his, obs_prop_vel, 5e-3)
        return loss

    def BarlowTwinsLoss(self, obs, obs_hist, obs_prop_vel, weight):
        obs, obs_hist = self.normalize(obs, obs_hist)
        obs_prop_vel = self.obs_vel_normalizer(obs_prop_vel)

        obs_hist_full = torch.cat([
            obs,
            obs_hist[:, :self.num_prop * (self.history_length - 1)],
        ], dim=1)
        b = obs.size()[0]

        # obs_hist = obs_hist[:,5:,:].reshape(b,-1)

        z1 = self.history_layer(obs_hist_full.reshape(b, -1))
        z2 = self.history_layer(obs_hist.reshape(b, -1))

        z1_l = self.latent_layer(z1)
        z1_v = self.vel_layer(z1)

        z2_l = self.latent_layer(z2)

        z1_l = self.projector(z1_l)
        z2_l = self.projector(z2_l)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        vel_loss = torch.nn.functional.mse_loss(z1_v, obs_prop_vel[:, self.num_prop:self.num_prop+3])  # 计算预测和真实速度的均方误差

        loss = on_diag + weight * off_diag + vel_loss

        return loss

    def normalize(self, obs, obs_hist):
        # 归一化单个观测（obs：(batch_size, num_proprio)）
        obs = self.obs_normalizer(obs)
        # 归一化观测历史
        batch_size = obs_hist.shape[0]
        obs_hist = self.obs_normalizer(obs_hist.reshape(-1, self.num_prop)).reshape(batch_size, -1)

        return obs, obs_hist

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

def build_mlp(input_dim, hidden_dims, output_dim, activation):
    """
    通用的MLP构建函数

    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        activation: 激活函数
    """
    layers = []

    # 添加第一层
    prev_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if i < len(hidden_dims) - 1:
            layers.append(activation)
        prev_dim = hidden_dim

    # 添加输出层
    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)

