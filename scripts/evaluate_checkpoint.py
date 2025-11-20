#!/usr/bin/env python3
"""
Evaluate a previously trained SAC model from checkpoint.

This script loads a saved model and VecNormalize stats, then evaluates
performance on various test scenarios.

Usage:
    # Basic evaluation on random states
    python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip

    # Evaluate on specific initial angle
    python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip \
        --theta0 -160

    # Multiple episodes
    python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip \
        --n-episodes 50

    # Quick grid analysis
    python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip \
        --grid-eval --grid-size 21

    # Compare with classical controller
    python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip \
        --compare-classical --theta0 -150
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import TimeLimit

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import (
    CartPendulumEnv,
    TrajectoryPlanner,
    rollout_rl_timed,
    rollout_classical_timed,
    create_state_grid,
    evaluate_state_grid,
    plot_basin_of_attraction,
    plot_timing_comparison,
    animate_trajectory,
    animate_comparison,
)


def load_checkpoint(model_path: str, vecnorm_path: Optional[str] = None, device: str = 'cpu'):
    """
    Load SAC model and VecNormalize from checkpoint.

    Args:
        model_path: Path to saved SAC model (.zip)
        vecnorm_path: Path to VecNormalize stats (.pkl)
                      If None, infers from model_path
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        model: Loaded SAC model
        vec_env: VecNormalize wrapped environment
    """
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Infer VecNormalize path if not provided
    if vecnorm_path is None:
        vecnorm_path = model_path.parent / "vecnormalize.pkl"
        if not vecnorm_path.exists():
            # Try alternative naming
            vecnorm_path = model_path.parent / "vec_normalize.pkl"
    else:
        vecnorm_path = Path(vecnorm_path).resolve()

    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found: {vecnorm_path}\n"
            f"Please specify path with --vecnorm argument"
        )

    print(f"Loading checkpoint:")
    print(f"  Model: {model_path}")
    print(f"  VecNormalize: {vecnorm_path}")
    print(f"  Device: {device}")

    # Load model
    model = SAC.load(str(model_path), device=device)

    # Create environment with same normalization
    def make_env():
        env = CartPendulumEnv()
        return TimeLimit(env, max_episode_steps=2000)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    print("✓ Checkpoint loaded successfully!\n")

    return model, vec_env


def evaluate_single_state(
    model,
    vec_env,
    theta0_deg: float = -160.0,
    max_seconds: float = 10.0,
    save_animation: Optional[str] = None
):
    """
    Evaluate model on a single initial state.

    Args:
        model: Loaded SAC model
        vec_env: VecNormalize environment
        theta0_deg: Initial angle in degrees
        max_seconds: Max simulation time
        save_animation: Optional path to save animation
    """
    print("="*80)
    print("Single State Evaluation")
    print("="*80)

    # Create initial state
    state = np.array([
        np.deg2rad(theta0_deg),  # theta
        0.0,                      # theta_dot
        0.0,                      # x
        0.0                       # x_dot
    ])

    print(f"\nInitial state: θ₀={theta0_deg:.1f}°, θ̇₀=0.0 rad/s, x₀=0.0 m, ẋ₀=0.0 m/s")
    print(f"Running rollout for up to {max_seconds}s...\n")

    # Run rollout
    trajectory, timing = rollout_rl_timed(model, vec_env, state, max_seconds=max_seconds)

    # Analyze results
    final_theta = trajectory['theta'].iloc[-1]
    final_theta_deg = np.rad2deg(final_theta)
    # Success: near upright (θ ≈ 0°) STANDARD RL CONVENTION
    theta_error = abs(final_theta)
    success = theta_error < np.deg2rad(10)

    print("Results:")
    print(f"  Duration: {trajectory['time'].iloc[-1]:.2f}s ({len(trajectory)} steps)")
    print(f"  Final angle: {final_theta_deg:+.2f}°")
    print(f"  Success: {'✓ YES' if success else '✗ NO'} (θ within 10° of upright)")
    print(f"  Max cart position: {trajectory['x'].abs().max():.3f} m (limit: 2.4m)")
    print(f"  Max control force: {trajectory['action'].abs().max():.2f} N (limit: 10N)")

    print(f"\nTiming statistics:")
    print(f"  Mean inference time: {timing['inference_time_mean_ms']:.3f} ms")
    print(f"  Std inference time:  {timing['inference_time_std_ms']:.3f} ms")
    print(f"  Min inference time:  {np.min(timing['per_step_times']):.3f} ms")
    print(f"  Max inference time:  {timing['inference_time_max_ms']:.3f} ms")

    # Create animation if requested
    if save_animation:
        print(f"\nCreating animation: {save_animation}")
        anim = animate_trajectory(trajectory, save_path=save_animation, fps=50)
        print("✓ Animation saved!")

    print("\n" + "="*80 + "\n")

    return trajectory, timing, success


def evaluate_multiple_episodes(
    model,
    vec_env,
    n_episodes: int = 50,
    curriculum_phase: str = "swingup"
):
    """
    Evaluate model on multiple random episodes.

    Args:
        model: Loaded SAC model
        vec_env: VecNormalize environment
        n_episodes: Number of episodes to run
        curriculum_phase: "swingup" or "stabilization"
    """
    print("="*80)
    print(f"Multiple Episodes Evaluation ({n_episodes} episodes)")
    print("="*80)
    print(f"Curriculum phase: {curriculum_phase}\n")

    results = []
    successes = 0

    # Create evaluation environment
    env = CartPendulumEnv(curriculum_phase=curriculum_phase)
    env = TimeLimit(env, max_episode_steps=2000)

    for ep in range(n_episodes):
        # Reset environment (random initial state based on curriculum)
        obs, _ = env.reset()
        state = env.unwrapped.get_state()

        # Run episode
        traj, timing = rollout_rl_timed(model, vec_env, state, max_seconds=10.0)

        # Check success (θ ≈ 0° for upright) STANDARD RL CONVENTION
        final_theta = traj['theta'].iloc[-1]
        theta_error = abs(final_theta)
        success = theta_error < np.deg2rad(10)
        successes += int(success)

        results.append({
            'episode': ep + 1,
            'theta_0': np.rad2deg(state[0]),
            'success': success,
            'final_theta': np.rad2deg(final_theta),
            'duration': traj['time'].iloc[-1],
            'inference_ms': timing['inference_time_mean_ms']
        })

        if (ep + 1) % 10 == 0:
            print(f"  Progress: {ep+1}/{n_episodes} episodes, "
                  f"success rate: {successes/(ep+1)*100:.1f}%")

    results_df = pd.DataFrame(results)

    # Summary statistics
    print(f"\nResults Summary:")
    print(f"  Success rate: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"  Mean final angle: {results_df['final_theta'].mean():.2f}°")
    print(f"  Mean episode duration: {results_df['duration'].mean():.2f}s")
    print(f"  Mean inference time: {results_df['inference_ms'].mean():.3f}ms")

    # Success by initial angle range
    print(f"\nSuccess breakdown by initial angle:")
    for angle_min in [-180, -120, -60, 0, 60, 120]:
        angle_max = angle_min + 60
        mask = (results_df['theta_0'] >= angle_min) & (results_df['theta_0'] < angle_max)
        if mask.sum() > 0:
            success_rate = results_df[mask]['success'].mean() * 100
            print(f"  θ₀ ∈ [{angle_min:+4d}°, {angle_max:+4d}°): "
                  f"{success_rate:5.1f}% ({results_df[mask]['success'].sum()}/{mask.sum()})")

    print("\n" + "="*80 + "\n")

    return results_df


def evaluate_with_classical_comparison(
    model,
    vec_env,
    theta0_deg: float = -150.0,
    save_animation: Optional[str] = None
):
    """
    Compare RL model with classical controller on same initial state.

    Args:
        model: Loaded SAC model
        vec_env: VecNormalize environment
        theta0_deg: Initial angle in degrees
        save_animation: Optional path to save comparison animation
    """
    print("="*80)
    print("RL vs Classical Comparison")
    print("="*80)

    state = np.array([np.deg2rad(theta0_deg), 0.0, 0.0, 0.0])

    # Display angle interpretation (STANDARD RL CONVENTION: θ=0 at top, θ=±180 at bottom)
    wrapped_deg = np.rad2deg(((state[0] + np.pi) % (2 * np.pi)) - np.pi)
    deviation_from_upright = abs(wrapped_deg)
    print(f"\nInitial state: θ₀={theta0_deg:.1f}°")
    print(f"  (Note: θ=0° is upright, θ=±180° is hanging down)")
    print(f"  Deviation from upright: {deviation_from_upright:.1f}°")

    # Create classical planner
    planner = TrajectoryPlanner(umax=20.0)

    # Run RL
    print("\nRunning RL controller...")
    traj_rl, timing_rl = rollout_rl_timed(model, vec_env, state, max_seconds=10.0)
    final_theta_rl = np.rad2deg(traj_rl['theta'].iloc[-1])
    # Success: near upright (θ ≈ 0°) STANDARD RL CONVENTION
    theta_error_rl = abs(traj_rl['theta'].iloc[-1])
    success_rl = theta_error_rl < np.deg2rad(10)

    # Run Classical
    print("Running Classical controller...")
    traj_classical, timing_classical = rollout_classical_timed(
        planner, vec_env, state, max_seconds=10.0
    )

    # Check if planning succeeded
    if not timing_classical.get('planning_success', False):
        print("  ⚠ WARNING: Classical planner failed to find trajectory!")
        print("  (BVP solver could not find a solution for this initial state)")

    final_theta_classical = np.rad2deg(traj_classical['theta'].iloc[-1]) if len(traj_classical) > 0 else theta0_deg
    # Success: near upright (θ ≈ 0°) STANDARD RL CONVENTION
    if len(traj_classical) > 0:
        theta_error_classical = abs(traj_classical['theta'].iloc[-1])
        success_classical = theta_error_classical < np.deg2rad(10)
    else:
        success_classical = False

    # Results
    print("\nResults:")
    print(f"  RL:")
    print(f"    Final angle: {final_theta_rl:+.2f}°")
    print(f"    Success: {'✓ YES' if success_rl else '✗ NO'}")
    print(f"    Inference time: {timing_rl['inference_time_mean_ms']:.3f}ms (mean)")

    print(f"  Classical:")
    print(f"    Final angle: {final_theta_classical:+.2f}°")
    print(f"    Success: {'✓ YES' if success_classical else '✗ NO'}")
    print(f"    Planning time: {timing_classical.get('initial_plan_time_ms', 0):.1f}ms")
    print(f"    Planning success: {'✓' if timing_classical.get('planning_success', False) else '✗'}")

    # Timing comparison
    if success_rl and success_classical:
        speedup = timing_classical.get('initial_plan_time_ms', 0) / timing_rl['inference_time_mean_ms']
        print(f"\n  RL is ~{speedup:.0f}× faster than classical planning!")

    # Create comparison animation
    if save_animation:
        print(f"\nCreating comparison animation: {save_animation}")
        anim = animate_comparison(traj_rl, traj_classical, save_path=save_animation, fps=50)
        print("✓ Animation saved!")

    print("\n" + "="*80 + "\n")

    return traj_rl, traj_classical, timing_rl, timing_classical


def evaluate_grid(
    model,
    vec_env,
    grid_size: int = 21,
    save_plots: bool = True,
    output_dir: str = "eval_results"
):
    """
    Evaluate model on a grid of initial states (Basin of Attraction).

    Args:
        model: Loaded SAC model
        vec_env: VecNormalize environment
        grid_size: Number of points per dimension (total = grid_size^2)
        save_plots: Whether to save visualization plots
        output_dir: Directory to save results
    """
    print("="*80)
    print(f"Grid Evaluation (Basin of Attraction)")
    print("="*80)

    n_theta = grid_size
    n_theta_dot = max(int(grid_size * 0.75), 11)  # Aspect ratio ~4:3

    print(f"\nGrid configuration:")
    print(f"  θ₀ points: {n_theta} (range: [-180°, 180°])")
    print(f"  θ̇₀ points: {n_theta_dot} (range: [-8, 8] rad/s)")
    print(f"  Total states: {n_theta * n_theta_dot}")
    print(f"  x₀ = 0.0 m, ẋ₀ = 0.0 m/s (fixed)\n")

    # Create grid
    states = create_state_grid(n_theta=n_theta, n_theta_dot=n_theta_dot)

    # Create classical planner for comparison
    planner = TrajectoryPlanner(umax=20.0)

    # Evaluate
    print("Evaluating grid (this may take several minutes)...")
    print("Progress updates every 50 states.\n")

    results = evaluate_state_grid(
        states=states,
        model=model,
        vec_env=vec_env,
        planner=planner,
        max_seconds=10.0,
        success_threshold_deg=10.0,
        progress_every=50
    )

    # Summary
    print(f"\nGrid Evaluation Summary:")
    print(f"  RL success rate: {results['rl_success'].mean()*100:.1f}% "
          f"({results['rl_success'].sum()}/{len(results)})")
    print(f"  Classical success rate: {results['classical_success'].mean()*100:.1f}% "
          f"({results['classical_success'].sum()}/{len(results)})")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "grid_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Create plots
    if save_plots:
        print("\nCreating visualizations...")

        # RL Basin of Attraction
        fig_rl = plot_basin_of_attraction(
            results, controller='rl', metric='success',
            title='RL (SAC) Basin of Attraction'
        )
        fig_rl.savefig(output_path / "boa_rl.png", dpi=150, bbox_inches='tight')
        plt.close(fig_rl)

        # Classical Basin of Attraction
        fig_classical = plot_basin_of_attraction(
            results, controller='classical', metric='success',
            title='Classical Control Basin of Attraction'
        )
        fig_classical.savefig(output_path / "boa_classical.png", dpi=150, bbox_inches='tight')
        plt.close(fig_classical)

        # Timing comparison
        fig_timing = plot_timing_comparison(results)
        fig_timing.savefig(output_path / "timing_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig_timing)

        print(f"✓ Plots saved to: {output_path}/")

    print("\n" + "="*80 + "\n")

    return results


def main(args):
    """Main evaluation script."""
    print("\n" + "="*80)
    print("SAC Checkpoint Evaluation")
    print("="*80 + "\n")

    # Load checkpoint
    model, vec_env = load_checkpoint(
        args.model,
        args.vecnorm,
        args.device
    )

    # Run requested evaluation modes
    if args.grid_eval:
        # Grid evaluation
        evaluate_grid(
            model, vec_env,
            grid_size=args.grid_size,
            save_plots=True,
            output_dir=args.output_dir
        )

    elif args.compare_classical:
        # RL vs Classical comparison
        save_anim = None
        if args.save_animation:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_anim = str(output_path / "comparison.mp4")

        evaluate_with_classical_comparison(
            model, vec_env,
            theta0_deg=args.theta0,
            save_animation=save_anim
        )

    elif args.n_episodes > 1:
        # Multiple episodes
        evaluate_multiple_episodes(
            model, vec_env,
            n_episodes=args.n_episodes,
            curriculum_phase=args.curriculum
        )

    else:
        # Single state evaluation
        save_anim = None
        if args.save_animation:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_anim = str(output_path / f"trajectory_theta{args.theta0:.0f}.mp4")

        evaluate_single_state(
            model, vec_env,
            theta0_deg=args.theta0,
            max_seconds=args.max_seconds,
            save_animation=save_anim
        )

    print("Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAC model from checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single state evaluation
  python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip --theta0 -160

  # Multiple episodes
  python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip --n-episodes 50

  # Grid analysis
  python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip --grid-eval

  # Compare with classical
  python scripts/evaluate_checkpoint.py --model runs/sac_train/phase2/sac_model.zip --compare-classical
        """
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved SAC model (.zip file)"
    )

    # Optional arguments
    parser.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="Path to VecNormalize stats (.pkl file). If not specified, infers from model path."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on (default: cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results (default: eval_results)"
    )

    # Evaluation modes
    parser.add_argument(
        "--theta0",
        type=float,
        default=-160.0,
        help="Initial angle in degrees for single state evaluation (default: -160)"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=10.0,
        help="Maximum simulation time in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes for multiple episode evaluation (default: 1)"
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default="swingup",
        choices=["swingup", "stabilization"],
        help="Curriculum phase for multiple episodes (default: swingup)"
    )
    parser.add_argument(
        "--grid-eval",
        action="store_true",
        help="Run grid evaluation (Basin of Attraction analysis)"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=21,
        help="Grid size for Basin of Attraction (default: 21)"
    )
    parser.add_argument(
        "--compare-classical",
        action="store_true",
        help="Compare with classical controller"
    )
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save trajectory animation(s) to output directory"
    )

    args = parser.parse_args()
    main(args)
