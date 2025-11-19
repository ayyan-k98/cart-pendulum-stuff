#!/usr/bin/env python3
"""
Evaluation script for comparing RL and classical control.

This script loads a trained SAC model and compares it against classical
trajectory optimization + LQR control on various test scenarios.

FAIRNESS: Both controllers receive properly scaled observations to ensure
fair comparison.

Usage:
    # Evaluate on angle sweep
    python scripts/evaluate.py \\
        --model runs/sac_train/phase2/sac_model.zip \\
        --vecnorm runs/sac_train/phase2/vecnormalize.pkl \\
        --scenario angle_sweep

    # Evaluate on specific initial conditions
    python scripts/evaluate.py \\
        --model runs/sac_train/phase2/sac_model.zip \\
        --vecnorm runs/sac_train/phase2/vecnormalize.pkl \\
        --scenario custom \\
        --angles 180 135 90 45
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import compare_controllers, plot_comparison
from src.utils import ensure_directory_exists


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate and compare RL vs classical control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model paths
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained SAC model (.zip)'
    )
    parser.add_argument(
        '--vecnorm', type=str, required=True,
        help='Path to VecNormalize stats (.pkl)'
    )

    # Evaluation scenario
    parser.add_argument(
        '--scenario', type=str, default='angle_sweep',
        choices=['angle_sweep', 'hanging', 'upright', 'custom'],
        help='Evaluation scenario'
    )
    parser.add_argument(
        '--angles', type=float, nargs='+',
        help='Custom angles (degrees) for custom scenario'
    )

    # Environment parameters
    parser.add_argument(
        '--c-theta', type=float, default=0.02,
        help='Angular friction coefficient'
    )
    parser.add_argument(
        '--c-x', type=float, default=0.05,
        help='Linear friction coefficient'
    )
    parser.add_argument(
        '--eval-substeps', type=int, default=10,
        help='RK4 substeps for evaluation'
    )
    parser.add_argument(
        '--max-seconds', type=float, default=20.0,
        help='Maximum episode duration (seconds)'
    )

    # Output
    parser.add_argument(
        '--out-dir', type=str, default='runs/evaluation',
        help='Output directory for plots and results'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Disable plotting'
    )

    return parser.parse_args(args)


def get_start_states(scenario: str, custom_angles: list = None) -> list:
    """
    Get initial states for evaluation scenario.

    Args:
        scenario: Scenario name
        custom_angles: Custom angles (degrees) for 'custom' scenario

    Returns:
        List of initial states [θ, θ̇, x, ẋ]
    """
    if scenario == 'angle_sweep':
        # Sweep from hanging down to various angles
        angles_deg = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        angles_rad = [np.deg2rad(a) for a in angles_deg]
        return [np.array([theta, 0.0, 0.0, 0.0]) for theta in angles_rad]

    elif scenario == 'hanging':
        # Just hanging down
        return [np.array([np.pi, 0.0, 0.0, 0.0])]

    elif scenario == 'upright':
        # Small perturbations from upright
        angles_deg = [-15, -10, -5, 0, 5, 10, 15]
        angles_rad = [np.deg2rad(a) for a in angles_deg]
        return [np.array([theta, 0.0, 0.0, 0.0]) for theta in angles_rad]

    elif scenario == 'custom':
        if custom_angles is None:
            raise ValueError("--angles required for custom scenario")
        angles_rad = [np.deg2rad(a) for a in custom_angles]
        return [np.array([theta, 0.0, 0.0, 0.0]) for theta in angles_rad]

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def main(args=None):
    """Main evaluation function."""
    args = parse_args(args)

    print("=" * 80)
    print("Cart-Pendulum Controller Comparison")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"VecNormalize: {args.vecnorm}")
    print(f"Scenario: {args.scenario}")
    print("=" * 80)

    # Check files exist
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.vecnorm):
        print(f"ERROR: VecNormalize not found: {args.vecnorm}")
        sys.exit(1)

    # Create output directory
    ensure_directory_exists(args.out_dir)

    # Get initial states
    start_states = get_start_states(args.scenario, args.angles)
    print(f"\nEvaluating {len(start_states)} initial conditions...")

    # Run comparison
    rl_trajectories, classical_trajectories = compare_controllers(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        start_states=start_states,
        c_theta=args.c_theta,
        c_x=args.c_x,
        eval_substeps=args.eval_substeps,
        max_seconds=args.max_seconds,
        verbose=True
    )

    # Plot results
    if not args.no_plot:
        print("\nGenerating plots...")
        for i, (rl_traj, classical_traj) in enumerate(zip(rl_trajectories, classical_trajectories)):
            theta_deg = np.rad2deg(start_states[i][0])
            title = f"RL vs Classical: Initial angle = {theta_deg:.0f}°"

            save_path = os.path.join(args.out_dir, f"comparison_{i:02d}.png")
            plot_comparison(rl_traj, classical_traj, title=title, save_path=save_path)
            plt.close()

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    from src.evaluation import compute_metrics

    rl_successes = sum(compute_metrics(traj)['success'] for traj in rl_trajectories)
    classical_successes = sum(compute_metrics(traj)['success'] for traj in classical_trajectories)

    print(f"RL success rate: {rl_successes}/{len(rl_trajectories)} ({100*rl_successes/len(rl_trajectories):.1f}%)")
    print(f"Classical success rate: {classical_successes}/{len(classical_trajectories)} ({100*classical_successes/len(classical_trajectories):.1f}%)")

    # Average metrics
    rl_metrics_list = [compute_metrics(traj) for traj in rl_trajectories]
    classical_metrics_list = [compute_metrics(traj) for traj in classical_trajectories]

    print("\nAverage Metrics:")
    print(f"{'Metric':<30} {'RL':>12} {'Classical':>12}")
    print("-" * 55)

    avg_rl_angle_error = np.mean([np.rad2deg(m['mean_angle_error']) for m in rl_metrics_list])
    avg_classical_angle_error = np.mean([np.rad2deg(m['mean_angle_error']) for m in classical_metrics_list])
    print(f"{'Mean angle error (°)':<30} {avg_rl_angle_error:>12.2f} {avg_classical_angle_error:>12.2f}")

    avg_rl_control = np.mean([m['control_effort'] for m in rl_metrics_list])
    avg_classical_control = np.mean([m['control_effort'] for m in classical_metrics_list])
    print(f"{'Control effort (N·s)':<30} {avg_rl_control:>12.1f} {avg_classical_control:>12.1f}")

    print("=" * 80)
    print(f"Results saved to: {args.out_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
