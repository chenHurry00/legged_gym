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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


class cmd:
    """Command class for robot control"""
    def __init__(self):
        self.vx = 1.0    # forward/backward velocity
        self.vy = 0.0    # lateral velocity  
        self.dyaw = 0.0  # turning velocity
        
        # Target values for smooth interpolation
        self.target_vx = 1.0
        self.target_vy = 0.0
        self.target_dyaw = 0.0
        
        # Smoothing parameters
        self.acceleration = 3.0  # acceleration rate
        self.max_speed = 2.0     # maximum speed
        self.max_turn_rate = 2.0 # maximum turn rate


class VisualFeedback:
    """Handles visual feedback display"""
    def __init__(self, gym, viewer):
        self.gym = gym
        self.viewer = viewer
        self.display_timer = 0
        self.display_interval = 0.1  # Update display every 0.1 seconds
        
    def update_display(self, cmd_obj, key_states, dt):
        """Update the visual display of command values"""
        self.display_timer += dt
        
        if self.display_timer >= self.display_interval:
            self.display_timer = 0
            
            # Create display text
            cmd_text = f"Commands: vx={cmd_obj.vx:.2f} vy={cmd_obj.vy:.2f} dyaw={cmd_obj.dyaw:.2f}"
            target_text = f"Targets: vx={cmd_obj.target_vx:.2f} vy={cmd_obj.target_vy:.2f} dyaw={cmd_obj.target_dyaw:.2f}"
            
            # Show active keys
            active_keys = [key.upper() for key, pressed in key_states.items() if pressed]
            key_text = f"Active keys: {', '.join(active_keys) if active_keys else 'None'}"
            
            # Display in console (since we can't easily add overlay text to Isaac Gym viewer)
            print(f"\r{cmd_text} | {target_text} | {key_text}", end="", flush=True)


class KeyboardController:
    """Handles keyboard input for robot control"""
    def __init__(self, cmd_obj, env):
        self.cmd = cmd_obj
        self.env = env
        self.gym = env.gym
        self.viewer = env.viewer
        
        # Key states - track which keys are currently pressed
        self.key_states = {
            'w': False,
            's': False, 
            'a': False,
            'd': False
        }
        
        # Visual feedback
        self.visual_feedback = VisualFeedback(self.gym, self.viewer) if self.viewer else None
        
        # Subscribe to keyboard events using gymapi constants
        self._setup_keyboard_events()
        
    def _setup_keyboard_events(self):
        """Setup keyboard event subscriptions"""
        if self.viewer is None:
            return
            
        # Import gymapi here to get key constants
        from isaacgym import gymapi
        
        # W/S for forward/backward
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "move_forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "move_backward") 
        # A/D for turning
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_right")
        
    def process_keyboard_events(self):
        """Process keyboard events and update command targets"""
        if self.viewer is None:
            return
            
        # Check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            # Handle key press/release based on evt.value
            key_pressed = evt.value > 0
            
            if evt.action == "move_forward":
                self.key_states['w'] = key_pressed
            elif evt.action == "move_backward":
                self.key_states['s'] = key_pressed
            elif evt.action == "turn_left":
                self.key_states['a'] = key_pressed
            elif evt.action == "turn_right":
                self.key_states['d'] = key_pressed
                    
        # Update target velocities based on key states
        self._update_targets()
        
    def update_commands_and_display(self, dt):
        """Update commands and visual display"""
        self.update_commands(dt)
        if self.visual_feedback:
            self.visual_feedback.update_display(self.cmd, self.key_states, dt)
        
    def _update_targets(self):
        """Update target velocities based on current key states"""
        # Forward/backward control (W/S keys)
        if self.key_states['w'] and not self.key_states['s']:
            self.cmd.target_vx = self.cmd.max_speed
        elif self.key_states['s'] and not self.key_states['w']:
            self.cmd.target_vx = -self.cmd.max_speed
        else:
            self.cmd.target_vx = 0.0
            
        # Turning control (A/D keys)  
        if self.key_states['a'] and not self.key_states['d']:
            self.cmd.target_dyaw = self.cmd.max_turn_rate
        elif self.key_states['d'] and not self.key_states['a']:
            self.cmd.target_dyaw = -self.cmd.max_turn_rate
        else:
            self.cmd.target_dyaw = 0.0
            
        # Lateral velocity stays at 0 (no A/D lateral movement in this implementation)
        self.cmd.target_vy = 0.0
        
    def update_commands(self, dt):
        """Update actual command values with smooth interpolation"""
        # Smooth interpolation towards target values
        alpha = min(1.0, self.cmd.acceleration * dt)
        
        self.cmd.vx += (self.cmd.target_vx - self.cmd.vx) * alpha
        self.cmd.vy += (self.cmd.target_vy - self.cmd.vy) * alpha  
        self.cmd.dyaw += (self.cmd.target_dyaw - self.cmd.dyaw) * alpha
        

def sim2sim(args):
    """Main simulation function with keyboard control"""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Initialize command controller
    robot_cmd = cmd()
    
    # Initialize keyboard controller if not in headless mode
    keyboard_controller = None
    if not args.headless and env.viewer is not None:
        keyboard_controller = KeyboardController(robot_cmd, env)
        print("Keyboard controls:")
        print("W/S - Forward/Backward")
        print("A/D - Turn Left/Right") 
        print("ESC - Quit")
    
    # Simulation loop
    robot_index = 0  # which robot to control
    
    for i in range(10 * int(env.max_episode_length)):
        # Process keyboard input and update commands
        if keyboard_controller is not None:
            keyboard_controller.process_keyboard_events()
            keyboard_controller.update_commands_and_display(env.dt)
        else:
            # If no keyboard controller, use default smooth update for commands
            alpha = min(1.0, robot_cmd.acceleration * env.dt)
            robot_cmd.vx += (robot_cmd.target_vx - robot_cmd.vx) * alpha
            robot_cmd.vy += (robot_cmd.target_vy - robot_cmd.vy) * alpha
            robot_cmd.dyaw += (robot_cmd.target_dyaw - robot_cmd.dyaw) * alpha
            
        # Set robot commands based on current command values
        # Apply commands to first robot only for this demo
        env.commands[robot_index, 0] = robot_cmd.vx   # linear velocity x
        env.commands[robot_index, 1] = robot_cmd.vy   # linear velocity y  
        env.commands[robot_index, 2] = robot_cmd.dyaw # angular velocity yaw
        
        # Get policy actions and step environment
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # Render the environment (this also processes keyboard events)
        env.render()
        
        # Periodic status updates for headless mode
        if args.headless and i % 100 == 0:
            print(f"Step {i}: Commands - vx: {robot_cmd.vx:.2f}, vy: {robot_cmd.vy:.2f}, dyaw: {robot_cmd.dyaw:.2f}")


if __name__ == '__main__':
    args = get_args()
    sim2sim(args)