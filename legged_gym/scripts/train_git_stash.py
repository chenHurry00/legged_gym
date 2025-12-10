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
import os
import sys
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

# 导入训练快照管理器（同目录）
try:
    from training_snapshot_manager import create_stash_based_snapshot
except ImportError:
    print("⚠️ 警告: 无法导入训练快照管理器，将跳过快照创建")
    create_stash_based_snapshot = None


def create_training_snapshot_if_needed(args):
    """在训练开始前创建快照"""
    if create_stash_based_snapshot is None:
        print("⚠️ 跳过训练快照创建")
        return

    print("=" * 60)
    print("🚀 正在创建训练快照...")
    print("🛡️ 使用git stash，不影响当前状态")
    print("=" * 60)

    # 使用任务名称作为快照名称
    task_name = getattr(args, 'task', 'unknown')

    try:
        success, snapshot_info = create_stash_based_snapshot(task_name=task_name)
        if success:
            print(f"✅ 快照创建成功: {snapshot_info['branch_name']}")
            if snapshot_info['repository_state']['has_changes']:
                print(f"📝 包含修改: 暂存{snapshot_info['repository_state']['staged_files_count']}个, "
                      f"修改{snapshot_info['repository_state']['modified_files_count']}个, "
                      f"新增{snapshot_info['repository_state']['untracked_files_count']}个")
            print(f"🌿 快照已保存到Git分支，使用 'git checkout {snapshot_info['branch_name']}' 查看")
            print(f"✅ 您当前的修改和状态未受影响")
        else:
            print("❌ 训练快照创建失败，但将继续训练")
    except Exception as e:
        print(f"⚠️ 快照创建过程中出现错误: {e}")
        print("🔄 继续训练...")

    print("=" * 60)
    print("🏃‍♂️ 开始训练...")
    print("=" * 60)


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()

    # 在训练开始前创建绝对安全的快照
    create_training_snapshot_if_needed(args)

    # 开始训练
    train(args)
