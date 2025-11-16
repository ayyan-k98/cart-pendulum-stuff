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
    analysis: Basin of Attraction and grid evaluation tools
    visualization: Publication-quality plotting and heatmaps
    rendering: Matplotlib animations for trajectory visualization
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
from .evaluation import (
    rollout_rl,
    rollout_classical,
    rollout_rl_timed,
    rollout_classical_timed,
    compare_controllers,
    compute_metrics
)
from .analysis import (
    create_state_grid,
    compute_success_metrics,
    evaluate_state_grid,
    summarize_results
)
from .visualization import (
    plot_basin_of_attraction,
    plot_timing_comparison,
    plot_success_comparison,
    plot_metric_histogram
)
from .rendering import (
    animate_trajectory,
    animate_comparison
)
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
    # Environment
    'CartPendulumEnv',
    # Classical control
    'TrajectoryPlanner',
    'compute_lqr_gain',
    # Training
    'train_sac',
    'finetune_sac',
    'TextProgressCallback',
    # Evaluation
    'rollout_rl',
    'rollout_classical',
    'rollout_rl_timed',
    'rollout_classical_timed',
    'compare_controllers',
    'compute_metrics',
    # Analysis
    'create_state_grid',
    'compute_success_metrics',
    'evaluate_state_grid',
    'summarize_results',
    # Visualization
    'plot_basin_of_attraction',
    'plot_timing_comparison',
    'plot_success_comparison',
    'plot_metric_histogram',
    # Rendering
    'animate_trajectory',
    'animate_comparison',
    # Utilities
    'state_to_obs',
    'obs_to_state',
    'normalize_obs_from_state',
    'compute_angle_from_obs',
    'wrap_angle',
    'compute_energy',
    'check_device_available'
]
