# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import math


class CausalSelfAttention(nn.Module):
    """Causal self-attention module for sequential data"""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim)
            mask: optional attention mask
        Returns:
            output: same shape as input
        """
        # Handle both (T, B, C) and (B, T, C) formats
        if x.dim() == 3 and x.size(0) < x.size(1):
            # Likely (T, B, C), transpose to (B, T, C)
            x = x.transpose(0, 1)
            transposed = True
        else:
            transposed = False

        B, T, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Transpose back if needed
        if transposed:
            y = y.transpose(0, 1)

        return self.proj(y)


class Memory(nn.Module):
    """Memory module handling RNN and optional attention"""

    def __init__(self, input_size, rnn_type='lstm', num_layers=1, hidden_size=256,
                 use_attention=False, attention_num_heads=4, attention_dropout=0.1):
        super().__init__()

        self.use_attention = use_attention
        self.hidden_size = hidden_size

        # RNN
        rnn_cls = nn.GRU if rnn_type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

        # Optional attention
        if use_attention:
            self.attention = CausalSelfAttention(hidden_size, attention_num_heads, attention_dropout)
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(hidden_size)
            # FFN after attention (helps with expressiveness)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(attention_dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(attention_dropout)
            )

            # Initialize attention weights carefully
            self._init_attention_weights()

    def _init_attention_weights(self):
        """Initialize attention weights to small values for stability"""
        for module in [self.attention, self.ffn]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param, gain=0.01)  # Small initialization
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, input, masks=None, hidden_states=None):
        """
        Args:
            input: (seq_len, batch, input_size) for batch mode or (1, batch, input_size) for inference
            masks: None for inference, tensor for batch mode
            hidden_states: None for inference (uses self.hidden_states), provided for batch mode
        """
        batch_mode = masks is not None

        if batch_mode:
            # Batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")

            # RNN forward
            out, _ = self.rnn(input, hidden_states)

            # Apply attention if enabled
            if self.use_attention:
                # Pre-norm + Attention + Residual
                normed = self.layer_norm1(out)
                attn_out = self.attention(normed)
                out = out + attn_out

                # Pre-norm + FFN + Residual
                normed = self.layer_norm2(out)
                ffn_out = self.ffn(normed)
                out = out + ffn_out

            # Unpad trajectories based on masks
            # This requires the unpad_trajectories function from rsl_rl
            try:
                from rsl_rl.utils import unpad_trajectories
                out = unpad_trajectories(out, masks)
            except ImportError:
                # Fallback: simple masking
                out = out * masks.unsqueeze(-1)
        else:
            # Inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0) if input.dim() == 2 else input,
                                               self.hidden_states)

            # Apply attention if enabled
            if self.use_attention:
                # Pre-norm + Attention + Residual
                normed = self.layer_norm1(out)
                attn_out = self.attention(normed)
                out = out + attn_out

                # Pre-norm + FFN + Residual
                normed = self.layer_norm2(out)
                ffn_out = self.ffn(normed)
                out = out + ffn_out

        return out

    def reset(self, dones=None):
        """Reset hidden states for done environments"""
        if self.hidden_states is None:
            return

        # When RNN is LSTM, hidden_states is a tuple (h, c)
        if isinstance(self.hidden_states, tuple):
            for hidden_state in self.hidden_states:
                if dones is None:
                    hidden_state.zero_()
                else:
                    hidden_state[..., dones, :] = 0.0
        else:
            # GRU case
            if dones is None:
                self.hidden_states.zero_()
            else:
                self.hidden_states[..., dones, :] = 0.0


class RNNAttentionActorCritic(nn.Module):
    """Base Actor-Critic without recurrence"""
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " +
                  str([key for key in kwargs.keys()]))
        super(RNNAttentionActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
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
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


class RNNAttentionActorCriticRecurrent(RNNAttentionActorCritic):
    """Actor-Critic with RNN, inheriting from base ActorCritic"""
    is_recurrent = True

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 rnn_type='lstm',
                 rnn_hidden_size=256,
                 rnn_num_layers=1,
                 init_noise_std=1.0,
                 use_attention=False,
                 attention_num_heads=2,
                 attention_dropout=0.1,
                 **kwargs):
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " +
                  str(list(kwargs.keys())))

        # Initialize parent with rnn_hidden_size as input
        super().__init__(num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

        # Create memory modules
        self.memory_a = Memory(num_actor_obs,
                               rnn_type=rnn_type,
                               num_layers=rnn_num_layers,
                               hidden_size=rnn_hidden_size,
                               use_attention=use_attention,
                               attention_num_heads=attention_num_heads,
                               attention_dropout=attention_dropout)

        self.memory_c = Memory(num_critic_obs,
                               rnn_type=rnn_type,
                               num_layers=rnn_num_layers,
                               hidden_size=rnn_hidden_size,
                               use_attention=use_attention,
                               attention_num_heads=attention_num_heads,
                               attention_dropout=attention_dropout)

        print(f"Actor RNN: {self.memory_a.rnn}")
        print(f"Critic RNN: {self.memory_c.rnn}")
        if use_attention:
            print(f"Using Causal Attention with {attention_num_heads} heads, dropout={attention_dropout}")

    def reset(self, dones=None):
        """Reset memory hidden states"""
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        """Sample actions during training or collection"""
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        """Deterministic actions for inference"""
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate state value"""
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        """Return current hidden states for actor and critic"""
        return self.memory_a.hidden_states, self.memory_c.hidden_states


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