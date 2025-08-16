# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.envs.base.legged_robot import LeggedRobot

class Balance(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    def compute_observations(self):
        """观测计算方法，处理平衡车的状态空间"""
        # 基础线速度（只取前进方向x）
        base_lin_vel_x = self.base_lin_vel[:, 0:1] * self.obs_scales.lin_vel
        
        # 基础角速度（取pitch和yaw方向）
        base_ang_vel_pitch = self.base_ang_vel[:, 1:2] * self.obs_scales.ang_vel  # pitch(y轴)
        base_ang_vel_yaw = self.base_ang_vel[:, 2:3] * self.obs_scales.ang_vel    # yaw(z轴)
        
        # 投影重力（表示当前倾斜角度）
        projected_gravity = self.projected_gravity
        
        # 命令（前进速度和yaw角速度）
        command_lin_vel_x = self.commands[:, 0:1] * self.commands_scale[0:1]  # x方向速度命令
        command_ang_vel_yaw = self.commands[:, 2:3] * self.commands_scale[2:3]  # yaw角速度命令
        
        # 轮子位置和速度
        dof_pos = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel = self.dof_vel * self.obs_scales.dof_vel
        
        # 上一步动作
        actions = self.actions
        
        # 组合所有观测
        self.obs_buf = torch.cat((
            base_lin_vel_x,         # 1 x方向
            base_ang_vel_pitch,     # 1 pitch方向
            base_ang_vel_yaw,       # 1 yaw方向
            projected_gravity,      # 3
            command_lin_vel_x,      # 1
            command_ang_vel_yaw,    # 1
            dof_pos,                # 2 (左右轮)
            dof_vel,                # 2 (左右轮)
            actions                 # 2
        ), dim=-1)
        
        # 添加噪声
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            
        return self.obs_buf
    
    def _get_noise_scale_vec(self, cfg):
        """为平衡车设置观测噪声缩放向量。
        
        观测结构:
        - base_lin_vel_x: 1维 (索引0)
        - base_ang_vel_pitch: 1维 (索引1)
        - base_ang_vel_yaw: 1维 (索引2)
        - projected_gravity: 3维 (索引3-5)
        - command_lin_vel_x: 1维 (索引6)
        - command_ang_vel_yaw: 1维 (索引7)
        - dof_pos: 2维 (索引8-9) (左右轮)
        - dof_vel: 2维 (索引10-11) (左右轮)
        - actions: 2维 (索引12-13)
        
        Args:
            cfg (Dict): 环境配置文件

        Returns:
            [torch.Tensor]: 用于缩放[-1, 1]均匀分布噪声的向量
        """
        # 创建与观测维度匹配的噪声向量 (14维)
        noise_vec = torch.zeros_like(self.obs_buf[0])
        
        # 从配置中获取噪声参数
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # 为每个观测分量设置噪声水平
        # 线速度 (索引0)
        noise_vec[0] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        
        # 角速度 (索引1-2)
        noise_vec[1:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        
        # 投影重力 (索引3-5)
        noise_vec[3:6] = noise_scales.gravity * noise_level
        
        # 命令 - 通常不添加噪声 (索引6-7)
        noise_vec[6:8] = 0.
        
        # 轮子位置 (索引8-9)
        noise_vec[8:10] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        
        # 轮子速度 (索引10-11)
        noise_vec[10:12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        
        # 上一步动作 - 通常不添加噪声 (索引12-13)
        noise_vec[12:14] = 0.
        
        return noise_vec
    
    def _reset_dofs(self, env_ids):
        """重写DOF重置方法，给予轻微的随机初始姿态，促进探索"""
        # 对车轮位置进行随机初始化
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.9, 1.1, (len(env_ids), self.num_dof), device=self.device)
        # 对车轮速度进行随机初始化
        self.dof_vel[env_ids] = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        
        # 更新物理引擎
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                             gymtorch.unwrap_tensor(self.dof_state),
                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """重写根状态重置方法，给予轻微的随机初始姿态"""
        # 基础位置和姿态
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # 添加轻微的随机旋转（围绕侧向轴）以便机器人需要自行平衡
        rand_euler = torch_rand_float(-0.15, 0.15, (len(env_ids), 3), device=self.device)
        rand_euler[:, 0] = 0  # 只在侧向轴添加随机性
        rand_euler[:, 2] = 0
        rand_quat = quat_from_euler_xyz(rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2])
        self.root_states[env_ids, 3:7] = rand_quat
        
        # 添加轻微的随机速度
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device)
        
        # 更新物理引擎
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reward_orientation(self):
        """重写姿态奖励，鼓励平衡车保持直立"""
        # 使用投影重力来衡量直立程度
        # 投影重力的x分量表示前后倾斜度
        forward_gravity = self.projected_gravity[:, 0]
        # 投影重力的y分量表示左右倾斜度
        lateral_gravity = self.projected_gravity[:, 1]
        
        # 计算与理想直立状态的差距
        up_reward = torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) / 0.15)
        
        return up_reward
    
    def _reward_tracking_ang_vel(self):
        """奖励yaw角速度跟踪，用于平衡车转向控制"""
        # 计算yaw角速度(z轴)的跟踪误差
        yaw_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        
        # 使用指数形式的奖励函数，误差越小奖励越大
        tracking_reward = torch.exp(-yaw_vel_error / self.cfg.rewards.tracking_sigma)
        
        return tracking_reward
    
    def _reward_tracking_lin_vel(self):
        """重写线速度跟踪奖励，适用于平衡车控制"""
        # 只关注前进/后退方向的速度匹配
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        
        # 使用指数衰减奖励，误差越小奖励越大
        tracking_reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        
        return tracking_reward