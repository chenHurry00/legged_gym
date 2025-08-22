# sim2sim.py - Interactive Robot Control

This script provides keyboard-controlled robot simulation with smooth command interpolation and real-time feedback.

## Features

- **Keyboard Control**: Use WASD keys to control robot movement
- **Smooth Interpolation**: Commands change smoothly with configurable acceleration
- **Visual Feedback**: Real-time display of current command values
- **Isaac Gym Integration**: Built on the existing legged_gym framework

## Usage

```bash
python legged_gym/scripts/sim2sim.py --task=<task_name>
```

### Example Commands

```bash
# Run with AnymalC rough terrain task
python legged_gym/scripts/sim2sim.py --task=anymal_c_rough

# Run with flat terrain
python legged_gym/scripts/sim2sim.py --task=anymal_c_flat

# Run in headless mode (no GUI)
python legged_gym/scripts/sim2sim.py --task=anymal_c_rough --headless
```

## Controls

| Key | Action |
|-----|--------|
| W | Move Forward |
| S | Move Backward |
| A | Turn Left |
| D | Turn Right |
| ESC | Quit Simulation |

## Command System

The script implements a `cmd` class with the following initial values:
- `vx = 1.0` - Forward/backward velocity
- `vy = 0.0` - Lateral velocity (not used in WASD control)
- `dyaw = 0.0` - Turning velocity

### Smooth Interpolation

Commands are smoothly interpolated towards target values using:
- `acceleration = 3.0` - Rate of change
- `max_speed = 2.0` - Maximum linear velocity
- `max_turn_rate = 2.0` - Maximum angular velocity

## Implementation Details

### Classes

1. **`cmd`**: Stores current and target command values
2. **`KeyboardController`**: Handles keyboard input and updates commands
3. **`VisualFeedback`**: Provides real-time display of command values

### Key Features

- **Non-blocking input**: Uses Isaac Gym's event system
- **Smooth transitions**: Commands change gradually to avoid jerky movement
- **Real-time feedback**: Shows current and target values
- **Framework integration**: Works with existing task registry system

## Requirements

- Isaac Gym
- PyTorch
- NumPy
- legged_gym environment

## Architecture

The script follows the same pattern as `play.py` but adds:
1. Interactive command control instead of policy-based actions
2. Keyboard event handling
3. Smooth command interpolation
4. Visual feedback system

## Troubleshooting

- **Isaac Gym not found**: Install Isaac Gym following the legged_gym installation guide
- **No keyboard response**: Ensure the viewer window has focus
- **Jerky movement**: Adjust acceleration parameter in the `cmd` class