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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    elif hasattr(actor_critic, 'projector'):
        exporter = PolicyExporterLP(actor_critic)
        path = os.path.join(path, 'policy_1.pt')
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

class PolicyExporterLP(torch.nn.Module):
    """
    用于JIT导出的策略包装器
    只包含推理所需的前向传播逻辑
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.num_prop = actor_critic.num_prop
        self.history_length = actor_critic.history_length

        # 导出必要的网络模块
        self.history_layer = actor_critic.history_layer
        self.latent_layer = actor_critic.latent_layer
        self.vel_layer = actor_critic.vel_layer
        self.actor = actor_critic.actor

        # 导出归一化器的统计量（running mean和running std）
        self.obs_normalizer = actor_critic.obs_normalizer

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 推理模式

        Args:
            observations: 形状为 (batch_size, num_prop * (history_length + 1))
                         包含当前观测和历史观测

        Returns:
            actions: 形状为 (batch_size, num_actions) 的动作
        """
        # 提取当前观测和历史观测
        obs_hist = observations[:, :self.num_prop * self.history_length]
        obs_prop = observations[:, :self.num_prop]
        batch_size = obs_prop.shape[0]

        # 归一化
        obs_prop_norm = self.obs_normalizer(obs_prop)
        obs_hist_norm = self.obs_normalizer(obs_hist.reshape(-1, self.num_prop))

        # 编码历史
        obs_hist_flat = obs_hist_norm.reshape(batch_size, -1)
        latent = self.history_layer(obs_hist_flat)
        z = self.latent_layer(latent)
        vel = self.vel_layer(latent)

        # Actor输入
        actor_input = torch.cat([obs_prop_norm, vel, z], dim=-1)

        # 输出动作
        actions = self.actor(actor_input)

        return actions

    def export(self, path, device='auto', verify=True):
        """
        导出策略为JIT格式

        Args:
            path: 保存路径
            device: 导出设备 ('auto', 'cpu', 'cuda', 'same')
                   - 'auto': 自动选择（推荐，默认CPU用于部署）
                   - 'cpu': 强制CPU（适合实机部署）
                   - 'cuda' 或 'cuda:0': 强制GPU
                   - 'same': 保持当前设备
            verify: 是否验证导出的模型
        """
        # 保存原始设备
        original_device = next(self.parameters()).device

        # 确定导出设备
        if device == 'auto':
            export_device = 'cpu'  # 默认用CPU，适合部署
            print("[INFO] Using 'cpu' for deployment compatibility")
        elif device == 'same':
            export_device = original_device
        else:
            export_device = device

        # 移动到目标设备
        self.to(export_device)
        self.eval()

        # 创建示例输入
        num_obs = self.num_prop * (self.history_length + 1)
        example_input = torch.zeros(1, num_obs).to(export_device)

        print(f"Exporting policy to JIT format...")
        print(f"Export device: {export_device}")
        print(f"Input shape: {example_input.shape}")
        print(f"Number of proprioceptive states: {self.num_prop}")
        print(f"History length: {self.history_length}")

        # 使用torch.jit.trace进行导出
        try:
            with torch.no_grad():
                # 先测试forward
                test_output = self.forward(example_input)
                print(f"Output shape: {test_output.shape}")

                # Trace
                traced_policy = torch.jit.trace(self, example_input)

            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            # 保存模型
            traced_policy.save(path)

            # 验证导出的模型
            if verify:
                self._verify_export(path, example_input, test_output, export_device)

        except Exception as e:
            print(f"✗ Export failed: {e}")
            raise
        finally:
            # 恢复原始设备
            if export_device != original_device:
                print(f"[INFO] Restoring model to original device: {original_device}")
                self.to(original_device)

    def export_cpu(self, path, verify=True):
        """
        导出为CPU版本（推荐用于实机部署）

        Args:
            path: 保存路径
            verify: 是否验证导出的模型
        """
        self.export(path, device='cpu', verify=verify)

    def export_gpu(self, path, verify=True):
        """
        导出为GPU版本

        Args:
            path: 保存路径
            verify: 是否验证导出的模型
        """
        self.export(path, device='cuda', verify=verify)

    def _verify_export(self, path, example_input, expected_output, device):
        """验证导出的模型"""
        print("\nVerifying exported model...")
        try:
            loaded_policy = torch.jit.load(path, map_location=device)
            loaded_policy.eval()

            with torch.no_grad():
                verify_output = loaded_policy(example_input)

            max_diff = torch.max(torch.abs(expected_output - verify_output)).item()
            mean_diff = torch.mean(torch.abs(expected_output - verify_output)).item()

            print(f"  Max difference: {max_diff:.2e}")
            print(f"  Mean difference: {mean_diff:.2e}")

            if max_diff < 1e-5:
                print("✓ Export validation passed!")
            elif max_diff < 1e-3:
                print("⚠ Warning: Small numerical difference detected (likely acceptable)")
            else:
                print("✗ Warning: Large output difference detected!")

        except Exception as e:
            print(f"✗ Verification failed: {e}")
