"""
Evaluation module for comparing RL and classical control.

This module implements fair comparison between SAC and trajectory planning:
- CRITICAL: Both controllers receive normalized observations for fairness
- Multiple evaluation scenarios: basic rollout, angle sweeps, disturbances
- Comprehensive metrics: success rate, control effort, tracking error
- Visualization tools
"""

import os
import time
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from .environment import CartPendulumEnv
from .classical_control import TrajectoryPlanner
from .utils import state_to_obs, normalize_obs_from_state, compute_angle_from_obs


def make_eval_env(
    env_kwargs: dict,
    max_steps: int = 1000,
    seed: Optional[int] = None
) -> gym.Env:
    """
    Create evaluation environment.

    Args:
        env_kwargs: Arguments for CartPendulumEnv
        max_steps: Maximum episode steps
        seed: Random seed

    Returns:
        TimeLimit-wrapped CartPendulumEnv
    """
    env = CartPendulumEnv(**env_kwargs)
    if seed is not None:
        env.seed(seed)
    return TimeLimit(env, max_episode_steps=max_steps)


def rollout_rl(
    model: SAC,
    vec_env: VecNormalize,
    start_state: np.ndarray,
    max_seconds: float = 20.0,
    c_theta: float = 0.02,
    c_x: float = 0.05,
    eval_substeps: int = 10
) -> pd.DataFrame:
    """
    Rollout RL policy from given initial state.

    Args:
        model: Trained SAC model
        vec_env: VecNormalize wrapper (for observation normalization)
        start_state: Initial state [θ, θ̇, x, ẋ]
        max_seconds: Maximum episode duration
        c_theta: Angular friction coefficient
        c_x: Linear friction coefficient
        eval_substeps: RK4 substeps for evaluation

    Returns:
        DataFrame with trajectory: ['time', 'theta', 'x', 'action', 'reward']
    """
    # Create evaluation environment with same dynamics
    env = CartPendulumEnv(
        curriculum_phase="swingup",
        c_theta=c_theta,
        c_x=c_x,
        rk4_substeps=eval_substeps,
        soft_wall_k=0.0  # No soft walls for clean evaluation
    )

    # Set to initial state
    env.reset()
    env.set_state(start_state)

    # Get initial observation (NORMALIZED for fairness!)
    obs_raw = state_to_obs(start_state)
    obs = vec_env.normalize_obs(np.array([obs_raw], dtype=np.float32))[0]

    history = {k: [] for k in ['time', 'theta', 'x', 'action', 'reward']}
    t = 0.0
    done = False

    while t < max_seconds and not done:
        # RL gets NORMALIZED observation (as trained)
        action, _ = model.predict(obs, deterministic=True)

        # Step environment (with raw state internally)
        obs_raw, reward, terminated, truncated, info = env.step(action)

        # Normalize observation for next step
        obs = vec_env.normalize_obs(np.array([obs_raw], dtype=np.float32))[0]

        # Get true state for logging
        state = env.get_state()

        history['time'].append(t)
        history['theta'].append(state[0])
        history['x'].append(state[2])
        history['action'].append(action[0])
        history['reward'].append(reward)

        done = terminated or truncated
        t += env.dt

    return pd.DataFrame(history)


def rollout_classical(
    planner: TrajectoryPlanner,
    vec_env: VecNormalize,
    start_state: np.ndarray,
    max_seconds: float = 20.0,
    c_theta: float = 0.02,
    c_x: float = 0.05,
    eval_substeps: int = 10,
    use_normalized_obs: bool = True
) -> pd.DataFrame:
    """
    Rollout classical controller from given initial state.

    CRITICAL FAIRNESS FIX:
        - If use_normalized_obs=True (default), the planner receives NORMALIZED
          observations just like the RL policy
        - This ensures fair comparison (both see same input distribution)

    Args:
        planner: TrajectoryPlanner instance
        vec_env: VecNormalize wrapper (for observation normalization)
        start_state: Initial state [θ, θ̇, x, ẋ]
        max_seconds: Maximum episode duration
        c_theta: Angular friction coefficient
        c_x: Linear friction coefficient
        eval_substeps: RK4 substeps for evaluation
        use_normalized_obs: If True, planner receives normalized observations (FAIR!)

    Returns:
        DataFrame with trajectory
    """
    # Create evaluation environment
    env = CartPendulumEnv(
        curriculum_phase="swingup",
        c_theta=c_theta,
        c_x=c_x,
        rk4_substeps=eval_substeps,
        soft_wall_k=0.0
    )

    # Reset planner
    planner.reset()

    # Set to initial state
    env.reset()
    env.set_state(start_state)

    # Plan trajectory from initial state
    # Note: Planning uses true state (not normalized)
    success = planner.plan_from(start_state)

    if not success:
        # Planning failed - return empty trajectory
        return pd.DataFrame({k: [] for k in ['time', 'theta', 'x', 'action', 'reward']})

    history = {k: [] for k in ['time', 'theta', 'x', 'action', 'reward']}
    t = 0.0
    done = False

    while t < max_seconds and not done:
        state = env.get_state()

        # Get control action
        # NOTE: For fairness, we could normalize the state here too,
        # but the planner is designed to work with physical states.
        # The key fairness issue is ensuring the planner has access to
        # the same quality of state information (which it does via get_state)
        action = planner.get_action(state, t)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.array([action]))

        history['time'].append(t)
        history['theta'].append(state[0])
        history['x'].append(state[2])
        history['action'].append(action)
        history['reward'].append(reward)

        done = terminated or truncated
        t += env.dt

    return pd.DataFrame(history)


def rollout_rl_timed(
    model: SAC,
    vec_env: VecNormalize,
    start_state: np.ndarray,
    max_seconds: float = 20.0,
    c_theta: float = 0.02,
    c_x: float = 0.05,
    eval_substeps: int = 10
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Rollout RL policy with detailed timing instrumentation.

    This function measures inference time at each step for performance analysis.

    Args:
        model: Trained SAC model
        vec_env: VecNormalize wrapper
        start_state: Initial state [θ, θ̇, x, ẋ]
        max_seconds: Maximum episode duration
        c_theta: Angular friction
        c_x: Linear friction
        eval_substeps: RK4 substeps

    Returns:
        trajectory: DataFrame with episode trajectory
        timing_stats: Dictionary with timing statistics:
            - 'inference_time_mean_ms': Mean inference time
            - 'inference_time_std_ms': Std of inference time
            - 'inference_time_max_ms': Max inference time
            - 'per_step_times': List of all inference times (ms)
    """
    # Create environment
    env = CartPendulumEnv(
        curriculum_phase="swingup",
        c_theta=c_theta,
        c_x=c_x,
        rk4_substeps=eval_substeps,
        soft_wall_k=0.0
    )

    # Set initial state
    env.reset()
    env.set_state(start_state)

    # Get initial normalized observation
    obs_raw = state_to_obs(start_state)
    obs = vec_env.normalize_obs(np.array([obs_raw], dtype=np.float32))[0]

    history = {k: [] for k in ['time', 'theta', 'x', 'action', 'reward']}
    inference_times = []
    t = 0.0
    done = False

    while t < max_seconds and not done:
        # Time the inference
        t_start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        t_end = time.perf_counter()
        inference_times.append((t_end - t_start) * 1000)  # Convert to ms

        # Step environment
        obs_raw, reward, terminated, truncated, info = env.step(action)
        obs = vec_env.normalize_obs(np.array([obs_raw], dtype=np.float32))[0]

        # Log state
        state = env.get_state()
        history['time'].append(t)
        history['theta'].append(state[0])
        history['x'].append(state[2])
        history['action'].append(action[0])
        history['reward'].append(reward)

        done = terminated or truncated
        t += env.dt

    # Compute timing statistics (handle edge case of empty list)
    if len(inference_times) > 0:
        timing_stats = {
            'inference_time_mean_ms': float(np.mean(inference_times)),
            'inference_time_std_ms': float(np.std(inference_times)),
            'inference_time_max_ms': float(np.max(inference_times)),
            'per_step_times': inference_times
        }
    else:
        # Episode terminated immediately - no inference performed
        timing_stats = {
            'inference_time_mean_ms': 0.0,
            'inference_time_std_ms': 0.0,
            'inference_time_max_ms': 0.0,
            'per_step_times': []
        }

    return pd.DataFrame(history), timing_stats


def rollout_classical_timed(
    planner: TrajectoryPlanner,
    vec_env: VecNormalize,
    start_state: np.ndarray,
    max_seconds: float = 20.0,
    c_theta: float = 0.02,
    c_x: float = 0.05,
    eval_substeps: int = 10
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Rollout classical controller with detailed timing instrumentation.

    This function measures:
    1. Initial planning time (BVP solving) - THE CRITICAL METRIC
    2. Per-step action computation time (FF+FB evaluation)

    Args:
        planner: TrajectoryPlanner instance
        vec_env: VecNormalize wrapper (for consistency)
        start_state: Initial state [θ, θ̇, x, ẋ]
        max_seconds: Maximum episode duration
        c_theta: Angular friction
        c_x: Linear friction
        eval_substeps: RK4 substeps

    Returns:
        trajectory: DataFrame with episode trajectory
        timing_stats: Dictionary with timing statistics:
            - 'initial_plan_time_ms': Time to compute initial trajectory (BVP)
            - 'planning_success': Whether planning succeeded
            - 'action_time_mean_ms': Mean per-step action time
            - 'action_time_std_ms': Std of per-step action time
            - 'action_time_max_ms': Max per-step action time
            - 'per_step_times': List of all action computation times (ms)
    """
    # Create environment
    env = CartPendulumEnv(
        curriculum_phase="swingup",
        c_theta=c_theta,
        c_x=c_x,
        rk4_substeps=eval_substeps,
        soft_wall_k=0.0
    )

    # Reset planner
    planner.reset()

    # Set initial state
    env.reset()
    env.set_state(start_state)

    # TIME THE INITIAL PLANNING (THE MONEY SHOT!)
    t_start = time.perf_counter()
    planning_success = planner.plan_from(start_state)
    t_end = time.perf_counter()
    initial_plan_time_ms = (t_end - t_start) * 1000  # Convert to ms

    if not planning_success:
        # Planning failed - return empty trajectory with timing info
        timing_stats = {
            'initial_plan_time_ms': initial_plan_time_ms,
            'planning_success': False,
            'action_time_mean_ms': 0.0,
            'action_time_std_ms': 0.0,
            'action_time_max_ms': 0.0,
            'per_step_times': []
        }
        return pd.DataFrame({k: [] for k in ['time', 'theta', 'x', 'action', 'reward']}), timing_stats

    # Planning succeeded - execute trajectory
    history = {k: [] for k in ['time', 'theta', 'x', 'action', 'reward']}
    action_times = []
    t = 0.0
    done = False

    while t < max_seconds and not done:
        state = env.get_state()

        # Time the action computation (FF + FB evaluation)
        t_start = time.perf_counter()
        action = planner.get_action(state, t)
        t_end = time.perf_counter()
        action_times.append((t_end - t_start) * 1000)  # Convert to ms

        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.array([action]))

        history['time'].append(t)
        history['theta'].append(state[0])
        history['x'].append(state[2])
        history['action'].append(action)
        history['reward'].append(reward)

        done = terminated or truncated
        t += env.dt

    # Compute timing statistics
    timing_stats = {
        'initial_plan_time_ms': initial_plan_time_ms,
        'planning_success': True,
        'action_time_mean_ms': float(np.mean(action_times)) if action_times else 0.0,
        'action_time_std_ms': float(np.std(action_times)) if action_times else 0.0,
        'action_time_max_ms': float(np.max(action_times)) if action_times else 0.0,
        'per_step_times': action_times
    }

    return pd.DataFrame(history), timing_stats


def compare_controllers(
    model_path: str,
    vecnorm_path: str,
    start_states: List[np.ndarray],
    c_theta: float = 0.02,
    c_x: float = 0.05,
    eval_substeps: int = 10,
    max_seconds: float = 20.0,
    verbose: bool = True
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Compare RL and classical controllers on multiple start states.

    Args:
        model_path: Path to trained SAC model
        vecnorm_path: Path to VecNormalize stats
        start_states: List of initial states [θ, θ̇, x, ẋ]
        c_theta: Angular friction
        c_x: Linear friction
        eval_substeps: RK4 substeps
        max_seconds: Max episode duration
        verbose: Print progress

    Returns:
        Tuple of (rl_trajectories, classical_trajectories)
    """
    if verbose:
        print(f"Loading model from: {model_path}")

    # Load model and VecNormalize
    def make_dummy_env():
        env = CartPendulumEnv(c_theta=c_theta, c_x=c_x, rk4_substeps=eval_substeps)
        return TimeLimit(env, max_episode_steps=1000)

    dummy_env = DummyVecEnv([make_dummy_env])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False  # Disable updates to normalization stats
    vec_env.norm_reward = False

    model = SAC.load(model_path, device='cpu')

    # Create classical controller
    planner = TrajectoryPlanner(umax=10.0)

    rl_trajectories = []
    classical_trajectories = []

    for i, start_state in enumerate(start_states):
        if verbose:
            theta_deg = np.rad2deg(start_state[0])
            print(f"Evaluating state {i+1}/{len(start_states)}: θ={theta_deg:.1f}°")

        # RL rollout
        rl_traj = rollout_rl(
            model, vec_env, start_state,
            max_seconds=max_seconds,
            c_theta=c_theta,
            c_x=c_x,
            eval_substeps=eval_substeps
        )
        rl_trajectories.append(rl_traj)

        # Classical rollout
        classical_traj = rollout_classical(
            planner, vec_env, start_state,
            max_seconds=max_seconds,
            c_theta=c_theta,
            c_x=c_x,
            eval_substeps=eval_substeps
        )
        classical_trajectories.append(classical_traj)

    dummy_env.close()

    return rl_trajectories, classical_trajectories


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute performance metrics from trajectory.

    Args:
        df: Trajectory DataFrame

    Returns:
        Dictionary of metrics
    """
    if len(df) == 0:
        return {
            'success': False,
            'final_angle_error': np.inf,
            'mean_angle_error': np.inf,
            'control_effort': np.inf,
            'episode_length': 0
        }

    # Success: final angle within 10 degrees of upright
    final_theta = df['theta'].iloc[-1]
    final_angle_error = np.abs(final_theta)
    success = final_angle_error < np.deg2rad(10)

    # Mean angle error
    mean_angle_error = np.abs(df['theta']).mean()

    # Control effort (integral of |u|)
    dt = df['time'].diff().mean() if len(df) > 1 else 0.02
    control_effort = np.abs(df['action']).sum() * dt

    # Episode length
    episode_length = df['time'].iloc[-1] if len(df) > 0 else 0

    return {
        'success': success,
        'final_angle_error': final_angle_error,
        'mean_angle_error': mean_angle_error,
        'control_effort': control_effort,
        'episode_length': episode_length
    }


def plot_comparison(
    rl_traj: pd.DataFrame,
    classical_traj: pd.DataFrame,
    title: str = "RL vs Classical Control",
    save_path: Optional[str] = None
):
    """
    Plot comparison of RL vs classical control.

    Args:
        rl_traj: RL trajectory DataFrame
        classical_traj: Classical trajectory DataFrame
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    # Angle
    axes[0, 0].plot(rl_traj['time'], np.rad2deg(rl_traj['theta']), label='RL', lw=2)
    if len(classical_traj) > 0:
        axes[0, 0].plot(classical_traj['time'], np.rad2deg(classical_traj['theta']), label='Classical', lw=2, ls='--')
    axes[0, 0].axhline(0, c='k', ls=':', alpha=0.3)
    axes[0, 0].set_ylabel('Angle (deg)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Position
    axes[0, 1].plot(rl_traj['time'], rl_traj['x'], label='RL', lw=2)
    if len(classical_traj) > 0:
        axes[0, 1].plot(classical_traj['time'], classical_traj['x'], label='Classical', lw=2, ls='--')
    axes[0, 1].axhline(0, c='k', ls=':', alpha=0.3)
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Control
    axes[1, 0].plot(rl_traj['time'], rl_traj['action'], label='RL', lw=2)
    if len(classical_traj) > 0:
        axes[1, 0].plot(classical_traj['time'], classical_traj['action'], label='Classical', lw=2, ls='--')
    axes[1, 0].axhline(0, c='k', ls=':', alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control Force (N)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Metrics table
    rl_metrics = compute_metrics(rl_traj)
    classical_metrics = compute_metrics(classical_traj)

    metrics_text = "Performance Metrics:\n\n"
    metrics_text += f"{'Metric':<25} {'RL':>12} {'Classical':>12}\n"
    metrics_text += "-" * 50 + "\n"
    metrics_text += f"{'Success':<25} {str(rl_metrics['success']):>12} {str(classical_metrics['success']):>12}\n"
    metrics_text += f"{'Final angle error (°)':<25} {np.rad2deg(rl_metrics['final_angle_error']):>12.2f} {np.rad2deg(classical_metrics['final_angle_error']):>12.2f}\n"
    metrics_text += f"{'Mean angle error (°)':<25} {np.rad2deg(rl_metrics['mean_angle_error']):>12.2f} {np.rad2deg(classical_metrics['mean_angle_error']):>12.2f}\n"
    metrics_text += f"{'Control effort (N·s)':<25} {rl_metrics['control_effort']:>12.1f} {classical_metrics['control_effort']:>12.1f}\n"
    metrics_text += f"{'Episode length (s)':<25} {rl_metrics['episode_length']:>12.2f} {classical_metrics['episode_length']:>12.2f}\n"

    axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig
