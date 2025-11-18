"""
Analysis utilities for Basin of Attraction and performance metrics.

This module provides tools for:
- Generating state grids for systematic evaluation
- Computing success metrics from trajectories
- Running full grid evaluations with both controllers
- Comparing controller performance across state space
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from .environment import CartPendulumEnv
from .classical_control import TrajectoryPlanner
from .evaluation import rollout_rl_timed, rollout_classical_timed


def create_state_grid(
    theta_range: Tuple[float, float] = (-math.pi, math.pi),
    theta_dot_range: Tuple[float, float] = (-3.0, 3.0),
    n_theta: int = 41,
    n_theta_dot: int = 31,
    x_fixed: float = 0.0,
    x_dot_fixed: float = 0.0
) -> List[np.ndarray]:
    """
    Generate a 2D grid of initial states for Basin of Attraction analysis.

    The grid spans (θ₀, θ̇₀) while keeping x₀=0 and ẋ₀=0 fixed.
    This is the most informative 2D slice of the full 4D state space.

    Args:
        theta_range: (min, max) for initial angle (radians)
        theta_dot_range: (min, max) for initial angular velocity (rad/s)
        n_theta: Number of theta grid points (~10° resolution with 41 points)
        n_theta_dot: Number of theta_dot grid points (~0.2 rad/s with 31 points)
        x_fixed: Fixed cart position (default: 0.0)
        x_dot_fixed: Fixed cart velocity (default: 0.0)

    Returns:
        List of initial states [θ, θ̇, x, ẋ]

    Example:
        >>> grid = create_state_grid(n_theta=41, n_theta_dot=31)
        >>> len(grid)
        1271  # 41 * 31 = 1271 states
    """
    thetas = np.linspace(theta_range[0], theta_range[1], n_theta)
    theta_dots = np.linspace(theta_dot_range[0], theta_dot_range[1], n_theta_dot)

    states = []
    for theta in thetas:
        for theta_dot in theta_dots:
            state = np.array([theta, theta_dot, x_fixed, x_dot_fixed], dtype=np.float64)
            states.append(state)

    return states


def compute_success_metrics(trajectory: pd.DataFrame, dt: float = 0.02) -> Dict[str, float]:
    """
    Compute performance metrics from a trajectory.

    Success Criteria:
        - Final angle within 10° of upright: |θ_final| < 0.1745 rad
        - Settles and stays there (doesn't oscillate back out)

    Metrics:
        - success: Boolean indicator of success
        - settling_time: Time to reach and stay within 10° threshold
        - final_angle_error: |θ_final| in radians
        - mean_angle_error: Mean |θ| over trajectory
        - control_effort: ∫|u|dt (integral of absolute control)
        - max_control: max|u| (peak control force)
        - episode_length: Total episode duration

    Args:
        trajectory: DataFrame with columns ['time', 'theta', 'x', 'action', 'reward']
        dt: Timestep (default: 0.02s)

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = compute_success_metrics(trajectory)
        >>> print(f"Success: {metrics['success']}, Settling time: {metrics['settling_time']:.2f}s")
    """
    if len(trajectory) == 0:
        return {
            'success': False,
            'settling_time': float('inf'),
            'final_angle_error': float('inf'),
            'mean_angle_error': float('inf'),
            'control_effort': float('inf'),
            'max_control': 0.0,
            'episode_length': 0.0
        }

    # Success threshold: 10 degrees = 0.1745 radians
    SUCCESS_THRESHOLD = np.deg2rad(10)

    # Final angle error
    final_theta = trajectory['theta'].iloc[-1]
    final_angle_error = abs(final_theta)

    # Check if final state is successful
    success = final_angle_error < SUCCESS_THRESHOLD

    # Compute settling time (time to reach and stay within threshold)
    settling_time = float('inf')
    if success:
        # Find first time it enters the success region
        in_region = np.abs(trajectory['theta']) < SUCCESS_THRESHOLD
        if in_region.any():
            # Find the last time it was outside before final success
            first_success_idx = np.where(in_region)[0][0]

            # Check if it stays in region from this point on
            stays_in = np.all(in_region[first_success_idx:])
            if stays_in:
                settling_time = trajectory['time'].iloc[first_success_idx]

    # Mean angle error
    mean_angle_error = np.abs(trajectory['theta']).mean()

    # Control effort (integral of |u|)
    control_effort = np.abs(trajectory['action']).sum() * dt

    # Max control
    max_control = np.abs(trajectory['action']).max()

    # Episode length
    episode_length = trajectory['time'].iloc[-1] if len(trajectory) > 0 else 0.0

    return {
        'success': bool(success),
        'settling_time': float(settling_time),
        'final_angle_error': float(final_angle_error),
        'mean_angle_error': float(mean_angle_error),
        'control_effort': float(control_effort),
        'max_control': float(max_control),
        'episode_length': float(episode_length)
    }


def evaluate_state_grid(
    model_path: str,
    vecnorm_path: str,
    state_grid: List[np.ndarray],
    max_seconds: float = 20.0,
    c_theta: float = 0.02,
    c_x: float = 0.05,
    eval_substeps: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate both controllers on entire state grid with timing instrumentation.

    This is the main workhorse function for Basin of Attraction analysis.

    For each state in the grid:
        1. Run RL controller with timing
        2. Run classical controller with timing
        3. Compute metrics for both
        4. Log all results

    Args:
        model_path: Path to trained SAC model
        vecnorm_path: Path to VecNormalize statistics
        state_grid: List of initial states to evaluate
        max_seconds: Maximum episode duration
        c_theta: Angular friction coefficient
        c_x: Linear friction coefficient
        eval_substeps: RK4 substeps for evaluation
        verbose: Print progress messages

    Returns:
        DataFrame with results for all states and both controllers.
        Columns:
            - theta_0, theta_dot_0, x_0, x_dot_0: Initial state
            - controller: 'rl' or 'classical'
            - success, settling_time, final_angle_error, etc.: Performance metrics
            - inference_time_mean_ms (RL) or initial_plan_time_ms (classical): Timing
            - ... additional timing metrics

    Example:
        >>> grid = create_state_grid(n_theta=21, n_theta_dot=21)
        >>> results = evaluate_state_grid(
        ...     model_path='runs/sac_train/phase2/sac_model.zip',
        ...     vecnorm_path='runs/sac_train/phase2/vecnormalize.pkl',
        ...     state_grid=grid
        ... )
        >>> results.to_csv('boa_results.csv')
    """
    if verbose:
        print(f"Evaluating {len(state_grid)} states with both controllers...")
        print(f"Total evaluations: {2 * len(state_grid)}")

    # Load RL model and VecNormalize
    if verbose:
        print(f"\nLoading RL model from: {model_path}")

    def make_dummy_env():
        env = CartPendulumEnv(c_theta=c_theta, c_x=c_x, rk4_substeps=eval_substeps)
        return TimeLimit(env, max_episode_steps=1000)

    dummy_env = DummyVecEnv([make_dummy_env])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load(model_path, device='cpu')

    # Create classical controller
    planner = TrajectoryPlanner(umax=10.0)

    # Store results
    results = []

    # Evaluate each state
    for i, start_state in enumerate(state_grid):
        if verbose and (i % 50 == 0 or i == len(state_grid) - 1):
            pct = 100.0 * (i + 1) / len(state_grid)
            theta_deg = np.rad2deg(start_state[0])
            theta_dot = start_state[1]
            print(f"Progress: {i+1}/{len(state_grid)} ({pct:.1f}%) | "
                  f"θ₀={theta_deg:6.1f}°, θ̇₀={theta_dot:5.2f} rad/s")

        # --- RL Evaluation ---
        rl_traj, rl_timing = rollout_rl_timed(
            model, vec_env, start_state,
            max_seconds=max_seconds,
            c_theta=c_theta,
            c_x=c_x,
            eval_substeps=eval_substeps
        )

        rl_metrics = compute_success_metrics(rl_traj)

        # Combine RL results
        rl_result = {
            'theta_0': start_state[0],
            'theta_dot_0': start_state[1],
            'x_0': start_state[2],
            'x_dot_0': start_state[3],
            'controller': 'rl',
            **rl_metrics,
            **rl_timing
        }
        results.append(rl_result)

        # --- Classical Evaluation ---
        classical_traj, classical_timing = rollout_classical_timed(
            planner, vec_env, start_state,
            max_seconds=max_seconds,
            c_theta=c_theta,
            c_x=c_x,
            eval_substeps=eval_substeps
        )

        classical_metrics = compute_success_metrics(classical_traj)

        # Combine classical results
        classical_result = {
            'theta_0': start_state[0],
            'theta_dot_0': start_state[1],
            'x_0': start_state[2],
            'x_dot_0': start_state[3],
            'controller': 'classical',
            **classical_metrics,
            **classical_timing
        }
        results.append(classical_result)

    # Clean up
    dummy_env.close()

    if verbose:
        print("\nEvaluation complete!")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Add degree versions of angles for convenience
    df['theta_0_deg'] = np.rad2deg(df['theta_0'])
    df['final_angle_error_deg'] = np.rad2deg(df['final_angle_error'])
    df['mean_angle_error_deg'] = np.rad2deg(df['mean_angle_error'])

    return df


def summarize_results(results_df: pd.DataFrame, verbose: bool = True) -> Dict[str, Dict]:
    """
    Compute summary statistics from grid evaluation results.

    Args:
        results_df: DataFrame from evaluate_state_grid
        verbose: Print summary to console

    Returns:
        Dictionary with summary statistics for each controller

    Example:
        >>> summary = summarize_results(results_df)
        >>> print(f"RL success rate: {summary['rl']['success_rate']:.1%}")
    """
    summary = {}

    for controller in ['rl', 'classical']:
        data = results_df[results_df['controller'] == controller]

        success_rate = data['success'].mean()
        mean_settling = data[data['success']]['settling_time'].mean()
        mean_control_effort = data['control_effort'].mean()

        summary[controller] = {
            'success_rate': success_rate,
            'n_successes': int(data['success'].sum()),
            'n_total': len(data),
            'mean_settling_time': mean_settling,
            'mean_control_effort': mean_control_effort,
            'mean_angle_error_deg': data['mean_angle_error_deg'].mean()
        }

        # Add timing stats
        if controller == 'rl':
            summary[controller]['mean_inference_time_ms'] = data['inference_time_mean_ms'].mean()
            summary[controller]['max_inference_time_ms'] = data['inference_time_max_ms'].max()
        else:  # classical
            summary[controller]['mean_planning_time_ms'] = data['initial_plan_time_ms'].mean()
            summary[controller]['max_planning_time_ms'] = data['initial_plan_time_ms'].max()
            summary[controller]['planning_success_rate'] = data['planning_success'].mean()

    if verbose:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        for controller in ['rl', 'classical']:
            stats = summary[controller]
            print(f"\n{controller.upper()}:")
            print(f"  Success rate: {stats['success_rate']*100:.1f}% "
                  f"({stats['n_successes']}/{stats['n_total']})")
            print(f"  Mean settling time: {stats['mean_settling_time']:.2f}s")
            print(f"  Mean control effort: {stats['mean_control_effort']:.1f} N·s")
            print(f"  Mean angle error: {stats['mean_angle_error_deg']:.2f}°")

            if controller == 'rl':
                print(f"  Mean inference time: {stats['mean_inference_time_ms']:.3f}ms")
                print(f"  Max inference time: {stats['max_inference_time_ms']:.3f}ms")
            else:
                print(f"  Mean planning time: {stats['mean_planning_time_ms']:.1f}ms")
                print(f"  Max planning time: {stats['max_planning_time_ms']:.1f}ms")
                print(f"  Planning success rate: {stats['planning_success_rate']*100:.1f}%")

    return summary
