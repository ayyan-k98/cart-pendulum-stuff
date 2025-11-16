"""
Cart-Pendulum Control: RL vs Classical Control Comparison

This package provides a complete implementation for training and evaluating
reinforcement learning (SAC) and classical control (trajectory optimization + LQR)
approaches to the cart-pendulum swing-up and stabilization problem.

Modules:
    environment: CartPendulumEnv - High-fidelity Gymnasium environment with RK4
    classical_control: TrajectoryPlanner - Optimal control baseline
    training: SAC training pipeline with curriculum learning
    evaluation: Fair comparison tools with normalized observations
    utils: Helper functions for state conversion and normalization

Example:
    >>> from src import CartPendulumEnv, train_sac
    >>> model_path, vecnorm_path = train_sac(total_steps=500_000)

Author: Cart-Pendulum Research Team
License: MIT
"""

from .environment import CartPendulumEnv
from .classical_control import TrajectoryPlanner, compute_lqr_gain
from .training import train_sac, finetune_sac, TextProgressCallback
from .utils import (
    state_to_obs,
    obs_to_state,
    normalize_obs_from_state,
    compute_angle_from_obs,
    wrap_angle,
    compute_energy,
    check_device_available
)

# Register environment with Gymnasium
try:
    import gymnasium as gym
    from gymnasium.envs.registration import register

    register(
        id='CartPendulum-v0',
        entry_point='src.environment:CartPendulumEnv',
        max_episode_steps=1000,
    )
except Exception:
    # Registration may fail in some contexts, that's ok
    pass

__version__ = '1.0.0'
__all__ = [
    'CartPendulumEnv',
    'TrajectoryPlanner',
    'compute_lqr_gain',
    'train_sac',
    'finetune_sac',
    'TextProgressCallback',
    'state_to_obs',
    'obs_to_state',
    'normalize_obs_from_state',
    'compute_angle_from_obs',
    'wrap_angle',
    'compute_energy',
    'check_device_available'
]
