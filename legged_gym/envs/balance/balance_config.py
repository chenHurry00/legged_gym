# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BalanceCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        '''
        线速度 (1): 只有前进方向(x)的线速度，因为平衡车无法横向移动
        角速度 (2): pitch角速度(倾斜)和yaw角速度(转向)
        投影重力 (3): 表示当前倾斜角度的三维向量
        命令 (2): 前进速度命令和yaw角速度命令
        轮子位置 (2): 左右轮的位置
        轮子速度 (2): 左右轮的速度
        上一步动作 (2): 之前施加的控制动作
        '''
        num_observations = 12  # 观测数量
        num_privileged_obs = None
        num_actions = 2  # 两个轮子的控制
        episode_length_s = 30  # 每个episode的时长(秒)

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.225]  # 起始高度
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始为直立姿态
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'right_wheel_joint': 0.0,
            'left_wheel_joint': 0.0
        }

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class control(LeggedRobotCfg.control):
        control_type = 'T'  # 使用扭矩控制
        action_scale = 1.0  # 根据车轮大小和电机能力调整
        decimation = 4

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/balance/urdf/balance.urdf'
        name = "balance"
        foot_name = "wheel"  # 轮子作为"足部"检测接触
        penalize_contacts_on = []  # 底盘接触地面应该被惩罚
        terminate_after_contacts_on = ["base_link"]  # 底盘接触地面应该终止
        self_collisions = 1  # 1禁用，0启用自碰撞

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.225  # 车轮半径+些许余量
        upright_target = [0, 0, 1]  # z轴朝上
        only_positive_rewards = False

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0001
            dof_pos_limits = 0
            orientation = -10.0  # 直立姿态奖励
            tracking_lin_vel = 2.0  # 前进速度跟踪
            tracking_ang_vel = 1.0  # yaw角速度跟踪奖励
            collision = -5.0  # 增加碰撞惩罚

            # 禁用不需要的奖励
            base_height = 0.0
            feet_air_time = 0.0
            feet_stumble = 0.0

class BalanceCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_balance'

        max_iterations = 1500 # number of policy updates
  