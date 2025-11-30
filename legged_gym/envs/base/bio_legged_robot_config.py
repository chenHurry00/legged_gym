from .legged_robot_config import LeggedRobotCfg

class BioLeggedRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # 增加观测维度。原有的观测维度通常为48左右（取决于具体机器人）。
        # 需要增加1维，用于归一化的能量箱状态。
        num_observations = 48 + 12 + 1
        num_privileged_obs = None # 如果使用特权观测，也需相应增加

    class bio_energetics:
        """
        生物能量学与疲劳模型参数配置
        """
        enable_fatigue = True  # 总开关

        # 核心 CP 模型参数
        critical_power = 200.0  # 临界功率阈值 (CP)
        w_prime_capacity = 3000.0  # [Joules] W' 能量箱总容量

        # 恢复动力学参数
        recovery_time_constant = 20.0  # 恢复时间常数 tau

        # 功率计算权重
        # P_total = w_mech * sum(|tau*vel|) + w_thermal * sum(tau^2)
        mechanical_work_weight = 1.0
        thermal_loss_weight = 0.5  # 模拟焦耳热 (I^2 R)，假设电阻归一化

        # 疲劳后果
        # 当 W' 耗尽时，最大力矩被限制到的比例 (0.0 - 1.0)
        fatigue_torque_scale = 0.3

        # 初始状态随机化
        # 在重置时，能量箱初始电量的随机范围 [min, max] * capacity
        initial_energy_range = [0.8, 1.0]

    class fatigue:
        enable = True
        # 3CC 模型参数
        F = 0.05  # 疲劳速率 (对应电机温升时间常数倒数)
        R = 0.001  # 基础恢复速率 (对应自然冷却)
        r = 10.0  # 休息恢复乘数 (主动冷却或对流冷却增强)
        dt = 0.02  # 疲劳模型的积分步长 (通常与控制频率一致)

    class metabolic:
        # 针对 Unitree Go1 / AK80-6 的估算参数
        # 假设 12 个电机。连续力矩 ~5Nm. CP = 12 * 5^2 = 300.
        cp_limit = 800

        # 能量容量 (Joules-proxy).
        # 允许全功率(23Nm)爆发约 5秒.
        # Max Power Proxy = 12 * 23^2 ≈ 6348.
        # Excess = 6048. Capacity = 6048 * 5 ≈ 30000.
        w_prime_total = 5000.0

        # 恢复时间常数 (秒)
        tau_recovery = 0.2

    class rewards (LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            # 禁用传统的简单力矩惩罚
            torques = -0.0

            # 新增生物能量奖励
            efficiency = 0.0  # 替代部分力矩惩罚
            energy_tank = 0.0  # 长期生存激励
            fatigue_penalty = -0.0  # 当能量小于一定阈值时的重罚（软终止）

            # 新的生物学奖励
            metabolic_integrity = -1.0  # W' 耗尽惩罚
            ballistic_swing = -0.5  # 摆动相力矩惩罚
            dynamic_clearance = -1.0  # 速度相关的高度势垒

            # 3CC
            penalty_3cc_sum = -5.0  # 惩罚总疲劳
            penalty_3cc_max = -20.0  # 严厉惩罚最大疲劳 (引导负载均衡)
            penalty_3cc_var = -10.0  # 惩罚疲劳值方差

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
