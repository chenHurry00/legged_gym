from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from .legged_robot import LeggedRobot
from .bio_legged_robot_config import BioLeggedRobotCfg

class BioLeggedRobot(LeggedRobot):

    def _init_buffers(self):
        # 调用父类初始化标准缓冲区 (dof_state, root_state, etc.)
        super()._init_buffers()

        # === 生物能量学缓冲区初始化 ===

        # 1. 能量箱状态 (W')
        # 形状: (num_envs, 1)
        # 初始值: 满电状态 (w_prime_capacity)
        self.energy_tank = torch.full(
            (self.num_envs, 1),
            self.cfg.bio_energetics.w_prime_capacity,
            device=self.device,
            dtype=torch.float
        )

        # 2. 瞬时功率监测
        # 用于记录每一步的功率，供 Reward 函数或日志使用
        self.instantaneous_power = torch.zeros(
            (self.num_envs, 1),
            device=self.device,
            dtype=torch.float
        )

        # 3. 疲劳因子 (Fatigue Factor)
        # 记录当前力矩衰减系数，用于调试可视化
        self.fatigue_factor = torch.ones(
            (self.num_envs, 1),
            device=self.device,
            dtype=torch.float
        )


    def _update_bio_energetics(self):
        """
        根据 CP 模型更新 W' 状态，计算瞬时功率，并处理恢复逻辑。
        此函数应在 post_physics_step 中调用。
        """
        params = self.cfg.bio_energetics
        dt = self.dt  # 仿真步长，通常为 0.02s 或 0.01s

        # --- 1. 计算瞬时功率 (P_t) ---
        # self.torques: (num_envs, num_dofs) - 实际施加的力矩
        # self.dof_vel: (num_envs, num_dofs) - 关节速度

        # 机械功率: sum(|tau * vel|)
        # 使用绝对值是因为生物肌肉做负功（刹车）也消耗能量（虽然效率不同）
        mech_power = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1, keepdim=True)

        # 热损耗代理: sum(tau^2)
        # 模拟电机绕组发热，与力矩平方成正比
        thermal_power = torch.sum(torch.square(self.torques), dim=1, keepdim=True)

        # 总功率加权求和
        total_power = (params.mechanical_work_weight * mech_power +
                       params.thermal_loss_weight * thermal_power)

        # 更新缓冲区供观测使用
        self.instantaneous_power[:] = total_power

        # --- 2. 能量箱动力学 (W' Dynamics) ---

        cp = params.critical_power
        w_cap = params.w_prime_capacity
        tau = params.recovery_time_constant

        # 判断每个环境是处于消耗状态还是恢复状态
        # depletion_mask: Boolean Tensor (num_envs, 1)
        depletion_mask = total_power > cp

        # 增量计算 delta_E
        delta_energy = torch.zeros_like(self.energy_tank)

        # A. 消耗逻辑 (Depletion)
        # 线性消耗: dE = - (P - CP) * dt
        delta_energy[depletion_mask] = -(total_power[depletion_mask] - cp) * dt

        # B. 恢复逻辑 (Recovery)
        # 指数恢复: dE = (W_cap - E_current) * (1 - exp(-dt/tau))
        # 只有当功率低于 CP 时才恢复
        recovery_mask = ~depletion_mask
        if torch.any(recovery_mask):
            # 计算潜在恢复量 (基于当前亏空)
            current_deficit = w_cap - self.energy_tank[recovery_mask]
            recovery_rate = 1.0 - torch.exp(torch.tensor(-dt / tau, device=self.device))
            delta_energy[recovery_mask] = current_deficit * recovery_rate

            # 可选：如果 P 越低恢复越快，可以引入 (CP - P) 因子调节 tau
            # 此处使用最简指数模型

        # --- 3. 更新状态与边界截断 ---
        self.energy_tank += delta_energy
        self.energy_tank = torch.clamp(self.energy_tank, 0.0, w_cap)

        # --- 4. 计算疲劳因子 ---
        # 定义疲劳映射：能量越低，力矩限制越严
        # 简单的线性映射：
        # Full tank -> factor = 1.0
        # Empty tank -> factor = fatigue_torque_scale
        normalized_energy = self.energy_tank / w_cap
        min_scale = params.fatigue_torque_scale

        self.fatigue_factor = min_scale + (1.0 - min_scale) * normalized_energy


    def _compute_torques(self, actions):
        # --- 原有逻辑：PD 控制器计算 ---
        # 处理标量或向量形式的 action_scale
        if isinstance(self.cfg.control.action_scale, (int, float)):
            # 标量形式
            actions_scaled = actions * self.cfg.control.action_scale
        else:
            # 向量形式 - 将 action_scale 转换为张量并与 actions 相乘
            action_scale_tensor = torch.tensor(self.cfg.control.action_scale,
                                               dtype=actions.dtype,
                                               device=actions.device)
            actions_scaled = actions * action_scale_tensor
        control_type = self.cfg.control.control_type

        if control_type == "P":
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.targets)
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        # --- 新增逻辑：疲劳驱动的动态力矩限制 ---

        # self.fatigue_factor 形状是 (num_envs, 1)
        # torques 形状是 (num_envs, num_dofs)
        # 使用 expand_as 进行广播
        current_scale = self.fatigue_factor.expand_as(torques)

        # 计算当前的动态最大力矩
        # self.torque_limits 是物理设定的电机绝对最大值
        dynamic_limits = self.torque_limits * current_scale

        # 硬截断 (Clamping)
        # 这模拟了电池电压下降或电机过热保护导致的输出能力下降
        torques = torch.clamp(torques, -dynamic_limits, dynamic_limits)

        return torques


    def compute_observations(self):
        # ... 计算 dof_pos, dof_vel, base_vel 等...

        # 归一化能量状态 (0.0 - 1.0)
        # 使用 1.0 代表满电，0.0 代表耗尽，这对神经网络更友好
        normalized_energy = self.energy_tank / self.cfg.bio_energetics.w_prime_capacity

        # 将能量状态拼接到观测张量末尾
        # 注意：必须确保 cfg.env.num_observations 已相应增加
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            normalized_energy  # <--- 新增维度
        ), dim=-1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        # 随机化初始电量
        # 让 Agent 场学会在"一开始就很累"的情况下生存
        # 范围例如 [0.5, 1.0] 倍容量
        if self.cfg.bio_energetics.enable_fatigue:
            init_range = self.cfg.bio_energetics.initial_energy_range  # e.g. [0.5, 1.0]
            min_e, max_e = init_range

            rand_levels = (
                    torch.rand(len(env_ids), 1, device=self.device) *
                    (max_e - min_e) +
                    min_e
            )

            self.energy_tank[env_ids] = rand_levels * self.cfg.bio_energetics.w_prime_capacity


    def _reward_efficiency(self):
        # 目标：最大化单位功率的速度 (Velocity per Watt)
        # 加上 epsilon (e.g., 20W) 作为基准代谢率，防止除以零，
        # 同时惩罚静止状态（静止时 v=0, P>0, reward=0;
        # 而由于存在基准代谢 epsilon，静止并不是最优解，快速通过才是）

        vel_norm = torch.norm(self.root_states[:, 7:10], dim=1)
        power = self.instantaneous_power.squeeze(-1)

        epsilon = 20.0  # 基准代谢率 / 静态功耗
        return vel_norm / (power + epsilon)

    def _reward_energy_tank(self):
        # 鼓励保持 W' 接近满值
        # 使用平方项，使得电量越低时，边际惩罚越大（Loss 梯度越大）
        normalized_energy = self.energy_tank / self.cfg.bio_energetics.w_prime_capacity

        return (torch.exp(2 * (normalized_energy - 1))).squeeze()

    def _reward_fatigue_penalty(self):
        """
        e: 剩余能量张量 (0 ~ 0.002 0.001对应10%开始急剧惩罚)
        k: 惩罚系数，越大惩罚越强
        """
        normalized_energy = self.energy_tank / self.cfg.bio_energetics.w_prime_capacity

        return (1e-3 / ((normalized_energy + 1e-3) ** 2)).squeeze()
