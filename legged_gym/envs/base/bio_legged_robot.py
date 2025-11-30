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

        # === 3CC初始化 ===

        self.muscle_states = torch.zeros(self.num_envs, self.num_dofs, 3,
                                         dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化：全部处于静息态 (MR=1.0, MA=0, MF=0)
        self.muscle_states[..., 0] = 1.0

        # === 能量箱初始化 ===

        # 初始化 W' 余额为满状态
        self.w_prime_bal = torch.full(
            (self.num_envs,),
            self.cfg.metabolic.w_prime_total,
            device=self.device,
            dtype=torch.float
        )
        # 记录上一时刻的力矩用于计算功率变化率
        self.last_torques = torch.zeros_like(self.torques)


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

    def _post_physics_step_callback(self):
        self._update_metabolic_state()
        super()._post_physics_step_callback()

    def _update_metabolic_state(self):
        """
        基于 Skiba (CP/W') 模型更新机器人的能量状态
        """
        # 1. 计算瞬时负载 (Proxy Power)
        # 使用力矩平方和作为热损耗的近似
        current_load = torch.sum(torch.square(self.torques), dim=1)

        # 2. 计算相对于 CP 的盈余/赤字
        cp = self.cfg.metabolic.cp_limit
        excess_load = current_load - cp

        # 3. 欧拉积分更新
        dt = self.dt

        # 消耗阶段 (P > CP)
        drain = torch.clamp(excess_load, min=0.0) * dt

        # 恢复阶段 (P < CP)
        # 使用线性恢复简化模型，或者指数模型：
        # recovery_rate = (cp - current_load) * (1 - exp(-dt/tau))
        # 这里使用简化的线性恢复，稳定性更好：
        recovery_capacity = torch.clamp(cp - current_load, min=0.0)
        recovery = (recovery_capacity / self.cfg.metabolic.tau_recovery) * dt

        # 更新并截断
        self.w_prime_bal = self.w_prime_bal - drain + recovery
        self.w_prime_bal = torch.clamp(self.w_prime_bal, 0.0, self.cfg.metabolic.w_prime_total)


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

        # --- 疲劳驱动的动态力矩限制 ---

        # self.fatigue_factor 形状是 (num_envs, 1)
        # torques 形状是 (num_envs, num_dofs)
        # 使用 expand_as 进行广播
        ###current_scale = self.fatigue_factor.expand_as(torques)

        # 计算当前的动态最大力矩
        # self.torque_limits 是物理设定的电机绝对最大值
        ###dynamic_limits = self.torque_limits * current_scale

        # 硬截断 (Clamping)
        # 这模拟了电池电压下降或电机过热保护导致的输出能力下降
        ###torques = torch.clamp(torques, -dynamic_limits, dynamic_limits)

        # --- 3CC 疲劳模型嵌入 ---
        if self.cfg.fatigue.enable:
            # A. 计算目标负载 (Target Load, TL)
            # 将力矩归一化到 ，假设 max_torque 对应 100% MVC
            torque_limits = self.torque_limits.unsqueeze(0)  # (1,12)
            target_load = torch.abs(torques) / torque_limits  # (N,12)
            target_load = torch.clamp(target_load, 0.0, 1.0)

            # B. 提取当前状态
            MR = self.muscle_states[..., 0]
            MA = self.muscle_states[..., 1]
            MF = self.muscle_states[..., 2]

            # C. 确定控制器 C(t) - 驱动 MR 流向 MA
            # 简单的比例控制：试图让 MA 追上 TL
            # 增益 k_recruit 决定了肌肉响应速度，如果不希望建模激活延迟，可设大一些
            k_recruit = 10.0
            recruitment_flow = k_recruit * (target_load - MA)

            # D. 确定恢复率 R_eff
            # 如果目标负载极低，认为在休息，启用 r 参数
            is_resting = target_load < 0.1
            R_eff = torch.where(is_resting,
                                self.cfg.fatigue.R * self.cfg.fatigue.r,
                                self.cfg.fatigue.R)

            # E. 微分方程求解 (Euler Integration)
            # dMA = C(t) - F*MA
            dMA = recruitment_flow - (self.cfg.fatigue.F * MA)

            # dMF = F*MA - R*eff*MF
            dMF = (self.cfg.fatigue.F * MA) - (R_eff * MF)

            # dMR = -C(t) + R_eff*MF
            dMR = -recruitment_flow + (R_eff * MF)

            # F. 更新状态
            dt = self.sim_params.dt
            MR_new = MR + dMR * dt
            MA_new = MA + dMA * dt
            MF_new = MF + dMF * dt

            # G. 归一化与裁剪 (防止数值漂移)
            total = MR_new + MA_new + MF_new
            self.muscle_states[..., 0] = torch.clamp(MR_new / total, 0, 1)
            self.muscle_states[..., 1] = torch.clamp(MA_new / total, 0, 1)
            self.muscle_states[..., 2] = torch.clamp(MF_new / total, 0, 1)

            # (可选) H. 性能限制：如果 MF 过高，是否强制削减力矩？
            # 用户想法："判断自身能输出的力矩"。
            # 可以乘以此系数模拟肌肉力竭： scale = 1.0 - MF
            torques = torques * (1.0 - self.muscle_states[..., 2])

        return torques


    def compute_observations(self):
        # ... 计算 dof_pos, dof_vel, base_vel 等...

        # 归一化能量状态 (0.0 - 1.0)
        # 使用 1.0 代表满电，0.0 代表耗尽，这对神经网络更友好
        normalized_energy = self.energy_tank / self.cfg.bio_energetics.w_prime_capacity + 0.001
        energy_level = (self.w_prime_bal / self.cfg.metabolic.w_prime_total).unsqueeze(1)

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
            self.muscle_states[..., 2], # <--- 3CC疲劳值
            energy_level  # <--- 能量池
            # normalized_energy  # <--- 疲劳驱动
        ), dim=-1)

    def compute_observations(self):
        """ Computes observations
        """
        energy_level = (self.w_prime_bal / self.cfg.metabolic.w_prime_total).unsqueeze(1)

        current_obs = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                 self.projected_gravity,
                                 self.commands[:, :3] * self.commands_scale,
                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                 self.dof_vel * self.obs_scales.dof_vel,
                                 self.actions,
                                 self.muscle_states[..., 2],  # <--- 3CC疲劳值
                                 energy_level  # <--- 能量池
                                 ), dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[:self.num_proprio]

        # update obs history
        self.obs_buf = torch.cat((current_obs[:, :self.num_proprio], self.obs_buf[:, :-self.num_proprio]),
                                 dim=-1)

    def compute_privileged_observations(self):
        """ Compute privileged observations.
          """
        energy_level = (self.w_prime_bal / self.cfg.metabolic.w_prime_total).unsqueeze(1)

        current_obs = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                 self.projected_gravity,
                                 self.commands[:, :3] * self.commands_scale,
                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                 self.dof_vel * self.obs_scales.dof_vel,
                                 self.actions,
                                 self.muscle_states[..., 2],  # <--- 3CC疲劳值
                                 energy_level  # <--- 能量池
                                 ), dim=-1)

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)

        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_proprio],
                                             self.base_lin_vel * self.obs_scales.lin_vel,), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat(
                (self.privileged_obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

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

        # 重置3CC参数
        self.muscle_states[..., 0] = 1.0
        self.muscle_states[..., 1] = 0.0
        self.muscle_states[..., 2] = 0.0

        #  重置能量箱 W' 余额为满状态
        self.w_prime_bal[env_ids] = self.cfg.metabolic.w_prime_total
        # 记录上一时刻的力矩用于计算功率变化率
        self.last_torques[env_ids] = torch.zeros_like(self.torques[env_ids])

    def _reward_penalty_3cc_max(self):
        fatigue = self.muscle_states[..., 2]  # (num_envs, num_dofs)
        max_fatigue = torch.max(fatigue, dim=1).values

        return max_fatigue

    def _reward_penalty_3cc_sum(self):
        fatigue = self.muscle_states[..., 2]  # (num_envs, num_dofs)
        sum_fatigue = torch.sum(fatigue, dim=1)

        return sum_fatigue

    def _reward_penalty_3cc_var(self):
        fatigue = self.muscle_states[..., 2]  # (num_envs, num_dofs)
        var_fatigue = torch.var(fatigue, dim=1, unbiased=False)

        return var_fatigue

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

    # ---------------------------------------------------------
    # 1. 代谢完整性奖励 (解决越障拒止)
    # ---------------------------------------------------------
    def _reward_metabolic_integrity(self):
        # 归一化电量
        w_norm = self.w_prime_bal / self.cfg.metabolic.w_prime_total

        # 设计软势垒：电量 > 40% 时无惩罚，< 40% 时指数惩罚
        threshold = 0.4
        safety_margin = w_norm - threshold

        # 使用 softplus 构造平滑的单边惩罚
        # 当 w_norm >> 0.4, -safety_margin 是负数, softplus -> 0
        # 当 w_norm << 0.4, -safety_margin 是正数, softplus -> 线性/指数增长
        penalty = torch.nn.functional.softplus(-safety_margin * 10.0)

        # 返回负值作为惩罚
        return penalty

    # ---------------------------------------------------------
    # 2. 弹道摆动奖励 (解决拖脚)
    # ---------------------------------------------------------
    def _reward_ballistic_swing(self):
        # 识别摆动相：接触力 Z 分量 < 1.0
        # contact_forces shape: (envs, feet, 3)
        contact_z = self.contact_forces[:, self.feet_indices, 2]
        is_swing = contact_z < 1.0

        # 获取各腿力矩
        leg_torques = self.torques.view(self.num_envs, 4, 3)
        # 计算每条腿的力矩平方和
        leg_effort = torch.sum(torch.square(leg_torques), dim=2)

        # 只惩罚摆动相的力矩
        # 逻辑：要在摆动相保持低力矩，必须在离地时给足初速度并抬高
        swing_penalty = leg_effort * is_swing.float()

        return torch.sum(swing_penalty, dim=1)

    # ---------------------------------------------------------
    # 3. 动态间隙势垒 (辅助抗拖曳)
    # ---------------------------------------------------------
    def _reward_dynamic_clearance(self):
        # 获取足端高度
        foot_z = self.rigid_body_states[:, self.feet_indices, 2]

        # 目标高度随速度增加: h = 0.02 + 0.1 * v_xy
        cmd_vel = torch.norm(self.commands[:, :2], dim=1).unsqueeze(1)
        target_h = 0.02 + 0.1 * cmd_vel

        # 摆动相掩码
        is_swing = self.contact_forces[:, self.feet_indices, 2] < 1.0

        # 惩罚项：仅当 foot_z < target_h 时产生
        height_error = target_h - foot_z
        penalty = torch.nn.functional.softplus(height_error * 20.0)

        return -torch.sum(penalty * is_swing.float(), dim=1)
