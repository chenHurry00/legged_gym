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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch


class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def  get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # 移除PD控制相关变量
    # target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_observations], dtype=np.double))

    count_lowlevel = 0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # 获取观测
        q, dq, quat, v, omega, gvec = get_obs(data)
        wheel_dq = dq[-cfg.env.num_actions:]  # 提取轮子速度

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # 创建观测向量 (12维)
            obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            # 1. 基础线速度x (1维)
            obs[0, 0] = v[0] * cfg.normalization.obs_scales.lin_vel
            # 2. 基础角速度pitch (1维)
            obs[0, 1] = omega[1] * cfg.normalization.obs_scales.ang_vel
            # 3. 基础角速度yaw (1维)
            obs[0, 2] = omega[2] * cfg.normalization.obs_scales.ang_vel
            # 4. 投影重力 (3维)
            obs[0, 3:6] = gvec
            # 5. 命令线速度x (1维)
            obs[0, 6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            # 6. 命令角速度yaw (1维)
            obs[0, 7] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            # 7. 轮子速度 (2维)
            obs[0, 8:10] = wheel_dq * cfg.normalization.obs_scales.dof_vel
            # 8. 上一步动作 (2维)
            obs[0, 10:12] = action

            # 应用观测剪裁
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            # 使用策略获取动作
            action_tensor = policy(torch.tensor(obs, dtype=torch.float32))
            action = action_tensor[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            # 直接将动作乘以比例因子作为扭矩
            tau = action * cfg.control.action_scale
            # 限制扭矩范围
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)

        # 直接设置扭矩，跳过PD控制
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        default='/home/yuchen/usetest/RL/legged_gym/logs/rough_balance/exported/policies/policy_1.pt',
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true',default='plane', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg:
        class env:
            num_observations = 12  # 总观测空间维度
            num_actions = 2  # 左右轮
            frame_stack = 1  # 不使用帧堆叠

        class normalization:
            # 定义与训练时一致的归一化参数
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                height = 5.0

            clip_observations = 100.0
            clip_actions = 100.0

        class control:
            # 控制参数
            control_type = 'T'  # 直接扭矩控制
            action_scale = 1.0

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/balance/mjcf/balance.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/balance/mjcf/balance.xml'
            sim_duration = 60.0
            dt = 0.005
            decimation = 4

        class robot_config:
            # 直接设置扭矩限制
            tau_limit = 200. * np.ones(2, dtype=np.double)


    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())