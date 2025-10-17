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


from collections import deque

import glfw  # Import glfw for keyboard input
import mujoco
import mujoco_viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from legged_gym import LEGGED_GYM_ROOT_DIR


class Cmd:
    def __init__(self):
        self.vx = 0.0  # Initialize with zero velocity
        self.vy = 0.0
        self.dyaw = 0.0
        self.vx_max = 1.0  # Maximum forward/backward velocity
        self.vy_max = 0.5  # Maximum lateral velocity
        self.dyaw_max = 0.5  # Maximum turning rate
        self.vx_step = 0.1  # Velocity change per key press
        self.vy_step = 0.1  # Lateral velocity change per key press
        self.dyaw_step = 0.1  # Turning rate change per key press

        # 跟踪按键状态
        self.w_pressed = False
        self.s_pressed = False
        self.a_pressed = False
        self.d_pressed = False
        self.q_pressed = False
        self.e_pressed = False

    def update_commands(self):
        """根据当前按键状态更新命令值"""
        # 重置命令
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0

        # 根据按键状态设置命令
        if self.w_pressed:
            self.vx = self.vx_max
        if self.s_pressed:
            self.vx = -self.vx_max
        if self.a_pressed:
            self.vy = self.vy_max
        if self.d_pressed:
            self.vy = -self.vy_max
        if self.q_pressed:
            self.dyaw = self.dyaw_max
        if self.e_pressed:
            self.dyaw = -self.dyaw_max


def keyboard_callback(window, key, scancode, action, mods):
    """Process keyboard input"""
    if key == glfw.KEY_W:
        if action == glfw.PRESS:
            cmd.w_pressed = True
            print("前进")
        elif action == glfw.RELEASE:
            cmd.w_pressed = False
            print("停止前进")

    elif key == glfw.KEY_S:
        if action == glfw.PRESS:
            cmd.s_pressed = True
            print("后退")
        elif action == glfw.RELEASE:
            cmd.s_pressed = False
            print("停止后退")

    elif key == glfw.KEY_A:
        if action == glfw.PRESS:
            cmd.a_pressed = True
            print("左移")
        elif action == glfw.RELEASE:
            cmd.a_pressed = False
            print("停止左移")

    elif key == glfw.KEY_D:
        if action == glfw.PRESS:
            cmd.d_pressed = True
            print("右移")
        elif action == glfw.RELEASE:
            cmd.d_pressed = False
            print("停止右移")

    elif key == glfw.KEY_Q:
        if action == glfw.PRESS:
            cmd.q_pressed = True
            print("左转")
        elif action == glfw.RELEASE:
            cmd.q_pressed = False
            print("停止左转")

    elif key == glfw.KEY_E:
        if action == glfw.PRESS:
            cmd.e_pressed = True
            print("右转")
        elif action == glfw.RELEASE:
            cmd.e_pressed = False
            print("停止右转")

    # 更新命令值
    cmd.update_commands()


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


def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)

    # 从base body获取四元数.orientation而不是传感器
    quat = data.qpos[3:7].astype(np.double)
    # 确保四元数格式为[w, x, y, z] -> [x, y, z, w]
    quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.double)

    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame

    # 从base body获取角速度而不是传感器
    # qvel[3:6]是世界坐标系的角速度，而四足机器人策略通常期望身体坐标系的角速度
    omega = r.apply(data.qvel[3:6], inverse=True).astype(np.double)

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

    # Register keyboard callback with the viewer's window
    glfw.set_key_callback(viewer.window, keyboard_callback)

    # PD控制参数
    kp = np.array([cfg.control.stiffness['joint']] * cfg.env.num_actions, dtype=np.double)
    kd = np.array([cfg.control.damping['joint']] * cfg.env.num_actions, dtype=np.double)

    default_joint_pos = np.array(list(cfg.init_state.default_joint_angles.values()), dtype=np.double)

    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_observations], dtype=np.double))

    count_lowlevel = 0

    # 步态固定参数
    gaits = [0. for _ in range(4)]
    gaits[0] = 2.5  # Frequency
    gaits[1] = 0.0  # Offset
    gaits[2] = 0.5  # Duration
    gaits[3] = 0.05  # Swing height
    gait_indices = np.remainder(
        cfg.sim_config.dt * gaits[0], 1.0
    )

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # Update GLFW events to process keyboard input
        glfw.poll_events()

        # 获取观测
        q, dq, quat, v, omega, gvec = get_obs(data)
        joint_q = q[7:]  # 提取关节位置 (前7个是base的自由度)
        joint_dq = dq[6:]  # 提取关节速度 (前6个是base的速度)

        frequencies = gaits[0]
        offsets = gaits[1]
        durations = gaits[2]
        # 更新步态相位
        gait_indices = np.remainder(
            gait_indices + cfg.sim_config.dt * frequencies, 1.0
        )
        # 生成时钟信号
        clock_inputs_sin = np.sin(2 * np.pi * gait_indices)
        clock_inputs_cos = np.cos(2 * np.pi * gait_indices)

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # 创建观测向量 (235维，与LeggedRobotCfg.env.num_observations一致)
            obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)

            # 2. 基础角速度 (3维)
            obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
            # 3. 投影重力 (3维)
            obs[0, 3:6] = gvec
            # 4. 命令 (3维)
            obs[0, 6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 7] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 8] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            # 5. 关节位置偏差 (12维)
            obs[0, 9:21] = (joint_q - default_joint_pos) * cfg.normalization.obs_scales.dof_pos
            # 6. 关节速度 (12维)
            obs[0, 21:33] = joint_dq * cfg.normalization.obs_scales.dof_vel
            # 7. 上一步动作 (12维)
            obs[0, 33:45] = action

            # 8. 高度测量 (187维)
            # 更准确的实现：模拟平坦地面的高度测量
            # 获取机器人基础位置
            base_pos = q[0:3]  # 基础位置 [x, y, z]
            # 简化的高度测量：假设地面是平的，高度为0
            # 在实际应用中，应该根据机器人位置和预定义的测量点网格计算高度
            heights = np.zeros(187, dtype=np.float32)
            # 机器人相对于地面的高度
            robot_height = base_pos[2]  # 机器人的z坐标
            # 根据训练时的归一化参数进行处理，与legged_robot.py中的实现一致
            heights = np.clip(robot_height - 0.5 - heights, -1, 1) * cfg.normalization.obs_scales.height_measurements
            # 填充观测向量
            # obs[0, 48:235] = heights
            # Clock inputs (sin, cos: 1D each)
            obs[0, 45] = clock_inputs_sin
            obs[0, 46] = clock_inputs_cos
            # Gait parameters (4D)
            obs[0, 47:51] = gaits

            # 应用观测剪裁
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            # 使用策略获取动作
            action_tensor = policy(torch.tensor(obs, dtype=torch.float32))
            action = action_tensor[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            # 计算目标关节位置
            target_q = action * cfg.control.action_scale + default_joint_pos

        # PD控制计算扭矩
        tau = pd_control(target_q, joint_q, kp, np.zeros_like(joint_dq), joint_dq, kd)

        # 设置控制扭矩
        if count_lowlevel >= 0:
            data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

        # Check if the viewer window is closed
        if viewer.is_alive is False:
            break

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=False,
                        default='/home/yuchen/usetest/RL/cyberdog_gym/legged_gym/legged_gym/logs/rough_Go2/exported/policies/policy_1.pt',
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default='plane', help='terrain or plane')
    args = parser.parse_args()


    class Sim2simCfg:
        class env:
            num_observations = 45 + 6 # 总观测空间维度 (3+3+3+3+12+12+12+187)
            num_actions = 12  # 12个关节
            frame_stack = 1  # 不使用帧堆叠

        class normalization:
            # 定义与训练时一致的归一化参数
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                height_measurements = 5.0

            clip_observations = 100.0
            clip_actions = 100.0

        class control:
            # PD控制参数
            control_type = 'P'  # 位置控制去
            stiffness = {'joint': 22.}  # [N*m/rad]
            damping = {'joint': 0.3}  # [N*m*s/rad]
            action_scale = 0.25

        class init_state:
            # 默认关节角度，与cyberdog_config.py中一致
            # 按照XML中关节的顺序排列
            # 代码里读取到的关节顺序和URDF定义的关节顺序不同，需要结合dof_names的输出校正xml定义的顺序
            default_joint_angles = {
                'FL_hip_joint': 0.1,  # [rad]
                'FL_thigh_joint': 0.8,  # [rad]
                'FL_calf_joint': -1.5,  # [rad]

                'FR_hip_joint': -0.1,  # [rad]
                'FR_thigh_joint': 0.8,  # [rad]
                'FR_calf_joint': -1.5,  # [rad]

                'RL_hip_joint': 0.1,  # [rad]
                'RL_thigh_joint': 1.,  # [rad]
                'RL_calf_joint': -1.5,  # [rad]

                'RR_hip_joint': -0.1,  # [rad]
                'RR_thigh_joint': 1.,  # [rad]
                'RR_calf_joint': -1.5,  # [rad]
            }

        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/mjcf/go2.xml'
            sim_duration = 60.0
            dt = 0.005
            decimation = 4


    # Initialize command object (replace the global cmd variable)
    cmd = Cmd()

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
