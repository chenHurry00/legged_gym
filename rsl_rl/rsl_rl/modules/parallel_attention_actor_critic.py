# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import math


class RunningMeanStd(nn.Module):
    """运行时均值标准差归一化"""

    def __init__(self, shape, eps=1e-8, clip_range=10.0):
        super().__init__()
        self.eps = eps
        self.clip_range = clip_range

        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('std', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1))

    def update(self, x):
        """更新running statistics"""
        with torch.no_grad():
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """使用Welford算法更新"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.std ** 2 * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_std = torch.sqrt(M2 / tot_count)

        # 防止std过小
        new_std = torch.clamp(new_std, min=self.eps)

        self.mean.copy_(new_mean)
        self.std.copy_(new_std)
        self.count.copy_(tot_count)

    def forward(self, x, update=False, clip=True):
        """归一化输入"""
        if update and self.training:
            self.update(x)

        normalized = (x - self.mean) / (self.std + self.eps)

        # Clip防止极端值
        if clip:
            normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)

        return normalized


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=50, dropout=0.0):
        super().__init__()
        # 使用更小的初始化
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.encoding[:, :seq_len, :]
        if self.dropout is not None:
            pos_enc = self.dropout(pos_enc)
        return x + pos_enc


class CausalSelfAttention(nn.Module):
    """因果自注意力"""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.register_buffer("bias", None)

        # 更保守的初始化
        self._init_weights()

    def _init_weights(self):
        """小心初始化attention权重"""
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, causal_mask=True):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * self.scale

        if causal_mask:
            if self.bias is None or self.bias.size(-1) < T:
                mask = torch.tril(torch.ones(T, T, device=x.device))
                self.register_buffer("bias", mask.view(1, 1, T, T))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)

        return self.proj_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, embed_dim, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)

        # 减小MLP ratio避免过拟合
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.ReLU(),  # 改用ReLU，比GELU更稳定
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """保守的权重初始化"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, causal_mask=True):
        x = x + self.attn(self.ln1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class HistoryEncoder(nn.Module):
    """历史观测编码器"""

    def __init__(self, obs_dim, num_frames, embed_dim=128,
                 num_heads=4, num_layers=2, dropout=0.1,
                 normalize_input=True, aggregation='last'):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.normalize_input = normalize_input
        self.aggregation = aggregation  # 'last', 'mean', 'max'

        # 输入归一化
        if normalize_input:
            self.input_norm = RunningMeanStd((obs_dim,))

        # 观测编码器：更简单的设计
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=num_frames, dropout=dropout)

        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)

        # 如果使用attention pooling
        if aggregation == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Softmax(dim=1)
            )

    def forward(self, obs_history, causal_mask=False, update_norm=False):
        """
        Args:
            obs_history: (B, T*obs_dim)
            causal_mask: 推理时设为False
            update_norm: 训练时设为True
        """
        B = obs_history.shape[0]

        # 重塑
        obs_seq = obs_history.reshape(B, self.num_frames, self.obs_dim)

        # 归一化
        if self.normalize_input:
            obs_flat = obs_seq.reshape(B * self.num_frames, self.obs_dim)
            obs_flat = self.input_norm(obs_flat, update=update_norm)
            obs_seq = obs_flat.reshape(B, self.num_frames, self.obs_dim)

        # 编码
        encoded_seq = self.obs_encoder(obs_seq)
        encoded_seq = self.pos_encoding(encoded_seq)

        # Transformer
        for block in self.transformer_blocks:
            encoded_seq = block(encoded_seq, causal_mask=causal_mask)

        encoded_seq = self.ln_f(encoded_seq)

        # 聚合
        if self.aggregation == 'last':
            output = encoded_seq[:, 0, :]  # 最新帧
        elif self.aggregation == 'mean':
            output = encoded_seq.mean(dim=1)
        elif self.aggregation == 'max':
            output = encoded_seq.max(dim=1)[0]
        elif self.aggregation == 'attention':
            weights = self.attention_pool(encoded_seq)  # (B, T, 1)
            output = (encoded_seq * weights).sum(dim=1)
        else:
            output = encoded_seq[:, 0, :]

        return output


class ParallelAttentionActorCritic(nn.Module):
    """并行Attention Actor-Critic (调试优化版)"""
    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_proprio,
                 num_scans=187,
                 embed_dim=128,
                 num_heads=4,
                 num_layers=2,
                 actor_hidden_dims=[256, 128],  # 减少参数
                 critic_hidden_dims=[256, 128],
                 scan_hidden_dims = [256, 128],
                 activation='elu',
                 dropout=0.0,  # 初期不用dropout
                 init_noise_std=1.0,
                 normalize_actor_input=True,
                 normalize_critic_input=True,
                 normalize_value=False,  # 可选：归一化value
                 history_aggregation='last',  # 历史聚合方式
                 **kwargs):

        if kwargs:
            print(f"Got unexpected arguments: {list(kwargs.keys())}")

        super().__init__()

        self.num_proprio = num_proprio
        self.num_frames = num_actor_obs // num_proprio
        self.num_scans = num_scans
        self.embed_dim = embed_dim
        self.normalize_actor_input = normalize_actor_input
        self.normalize_critic_input = normalize_critic_input
        self.normalize_value = normalize_value

        # ==================== Actor ====================
        self.actor_history_encoder = HistoryEncoder(
            obs_dim=num_proprio,
            num_frames=self.num_frames,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            normalize_input=normalize_actor_input,
            aggregation=history_aggregation
        )

        if normalize_actor_input:
            self.prop_input_norm = RunningMeanStd((self.num_proprio,))
        # Actor MLP
        activation_fn = get_activation(activation)
        actor_layers = []
        prev_dim = embed_dim + self.num_proprio
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn,
            ])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # 正交初始化actor
        self._init_actor_weights()

        # ==================== Critic ====================
        if normalize_critic_input:
            self.critic_input_norm = RunningMeanStd((num_critic_obs,))

        scan_layers = []
        prev_dim = self.num_scans
        for hidden_dim in scan_hidden_dims:
            scan_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn
            ])
            prev_dim = hidden_dim
        self.scan_layer = nn.Sequential(*scan_layers)

        critic_layers = []
        prev_dim = num_critic_obs - self.num_scans + scan_hidden_dims[-1]
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # 正交初始化critic
        self._init_critic_weights()

        # Value归一化（可选）
        if normalize_value:
            self.value_norm = RunningMeanStd((1,))

        # 动作噪声
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        self._print_architecture(num_critic_obs)

    def _init_actor_weights(self):
        """正交初始化Actor - 这很重要！"""
        for module in self.actor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)  # 小的gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_critic_weights(self):
        """正交初始化Critic"""
        for i, module in enumerate(self.critic.modules()):
            if isinstance(module, nn.Linear):
                # 最后一层用更小的gain
                gain = 0.01 if i == len(list(self.critic.modules())) - 1 else 1.0
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _print_architecture(self, num_critic_obs):
        actor_params = sum(p.numel() for p in self.actor_history_encoder.parameters()) + \
                       sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())

        print("\n" + "=" * 70)
        print("Parallel Attention Actor-Critic (Debugging Optimized)")
        print("=" * 70)
        print(f"Actor:")
        print(f"  History: {self.num_frames} frames × {self.num_proprio} dim")
        print(f"  Normalize input: {self.normalize_actor_input}")
        print(f"  Embed: {self.embed_dim}, Heads: {self.actor_history_encoder.transformer_blocks[0].attn.num_heads}")
        print(f"  Layers: {len(self.actor_history_encoder.transformer_blocks)}")
        print(f"  Aggregation: {self.actor_history_encoder.aggregation}")
        print(f"  Params: {actor_params:,}")
        print(f"\nCritic:")
        print(f"  Input: {num_critic_obs} dim (single frame)")
        print(f"  Normalize input: {self.normalize_critic_input}")
        print(f"  Normalize value: {self.normalize_value}")
        print(f"  Params: {critic_params:,}")
        print(f"\nTotal: {actor_params + critic_params:,} params")
        print(f"Action std: {self.std.data[0]:.3f}")
        print("=" * 70 + "\n")

    def reset(self, dones=None):
        pass

    def update_distribution(self, observations, update_norm=False):
        encoded = self.actor_history_encoder(
            observations, causal_mask=False, update_norm=update_norm
        )

        prop = observations[:, :self.num_proprio]
        if self.normalize_actor_input:
            prop = self.prop_input_norm(prop)

        input = torch.cat((prop, encoded), dim=1)
        mean = self.actor(input)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, update_norm=False, **kwargs):
        self.update_distribution(observations, update_norm=update_norm)
        return self.distribution.sample()

    def act_inference(self, observations):
        encoded = self.actor_history_encoder(
            observations, causal_mask=False, update_norm=False
        )

        prop = observations[:, :self.num_proprio]
        if self.normalize_actor_input:
            prop = self.prop_input_norm(prop)

        input = torch.cat((prop, encoded), dim=1)
        return self.actor(input)

    def evaluate(self, critic_observations, update_norm=False, **kwargs):
        if self.normalize_critic_input:
            critic_observations = self.critic_input_norm(
                critic_observations, update=update_norm
            )

        prop_vel = critic_observations[:, :-self.num_scans]
        scan = critic_observations[:, self.num_proprio+3:]

        scan_hidden_layer = self.scan_layer(scan)

        input = torch.cat((prop_vel, scan_hidden_layer), dim=1)
        value = self.critic(input)

        # Value归一化（逆变换）
        if self.normalize_value and update_norm:
            self.value_norm.update(value)

        return value

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)


def get_activation(act_name):
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "crelu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "gelu": nn.GELU(),
    }
    return activations.get(act_name.lower(), nn.ELU())