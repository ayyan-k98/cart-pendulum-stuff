#!/usr/bin/env python3
"""
Demonstration script for cart-pendulum visualization capabilities.

This script shows how to use:
1. Real-time pygame rendering during rollouts
2. Post-hoc matplotlib animations from saved trajectories
3. Comparison animations (RL vs Classical)

Usage:
    # Run with default settings (requires trained model)
    python scripts/visualize_demo.py

    # Specify custom model path
    python scripts/visualize_demo.py --model runs/sac_train/phase2/sac_model.zip

    # Save animations to files
    python scripts/visualize_demo.py --save-animations

Author: Cart-Pendulum Research Team
License: MIT
"""

import argparse
import os
import numpy as np
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src import (
    CartPendulumEnv,
    TrajectoryPlanner,
    rollout_rl_timed,
    rollout_classical_timed,
    animate_trajectory,
    animate_comparison
)


def demo_pygame_rendering(vec_env, model, start_state, max_seconds=10.0):
    """
    Demonstrate real-time pygame rendering during a rollout.

    Args:
        vec_env: VecNormalize wrapped environment
        model: Trained SAC model
        start_state: Initial state [θ, θ̇, x, ẋ]
        max_seconds: Maximum simulation time
    """
    print("\n" + "="*80)
    print("DEMO 1: Real-Time Pygame Rendering")
    print("="*80)
    print("\nThis will open a pygame window showing the cart-pendulum in real-time.")
    print("The window will display:")
    print("  - Cart (blue rectangle) on rail")
    print("  - Pendulum (red line with mass)")
    print("  - Current state values (θ, θ̇, x, ẋ)")
    print("  - Success indicator (UPRIGHT when |θ| < 10°)")
    print("\nCreating environment with render_mode='human'...")

    # Create environment with pygame rendering
    def make_render_env():
        env = CartPendulumEnv(render_mode='human')  # Enable pygame rendering
        return TimeLimit(env, max_episode_steps=1000)

    render_vec_env = DummyVecEnv([make_render_env])

    # Load normalization stats from training
    render_vec_env = VecNormalize.load(
        str(Path(args.vecnorm).resolve()),
        render_vec_env
    )
    render_vec_env.training = False
    render_vec_env.norm_reward = False

    # Set initial state
    env = render_vec_env.envs[0].unwrapped
    env.state = start_state.copy()

    # Run episode with rendering
    obs = render_vec_env.reset()
    done = False
    t = 0.0
    step_count = 0

    print(f"\nRunning rollout from θ₀={np.rad2deg(start_state[0]):.1f}°...")
    print("Close the pygame window to continue to next demo.\n")

    while t < max_seconds and not done:
        # Get action from policy
        action, _ = model.predict(obs, deterministic=True)

        # Step environment (this will call env.render() automatically)
        obs, reward, done, info = render_vec_env.step(action)

        t += env.dt
        step_count += 1

    print(f"Rollout complete: {step_count} steps, {t:.2f}s simulated")
    render_vec_env.close()


def demo_matplotlib_animation(vec_env, model, planner, start_state, save_path=None):
    """
    Demonstrate matplotlib post-hoc animation from trajectory data.

    Args:
        vec_env: VecNormalize wrapped environment
        model: Trained SAC model
        planner: TrajectoryPlanner for classical control
        start_state: Initial state
        save_path: Optional path to save animation (MP4 or GIF)
    """
    print("\n" + "="*80)
    print("DEMO 2: Matplotlib Post-Hoc Animation")
    print("="*80)
    print("\nThis creates publication-quality animations from saved trajectory data.")
    print("Features:")
    print("  - Side-by-side cart animation + angle plot")
    print("  - Can save to MP4 (requires ffmpeg) or GIF")
    print("  - No pygame dependency needed")
    print("\nRunning RL rollout to collect trajectory data...")

    # Collect trajectory
    traj_rl, timing = rollout_rl_timed(
        model, vec_env, start_state,
        max_seconds=10.0
    )

    print(f"Collected {len(traj_rl)} timesteps")
    print(f"Success: {abs(traj_rl['theta'].iloc[-1]) < np.deg2rad(10)}")
    print(f"Final angle: {np.rad2deg(traj_rl['theta'].iloc[-1]):.1f}°")

    # Create animation
    print("\nCreating matplotlib animation...")
    if save_path:
        print(f"Saving to: {save_path}")
        anim = animate_trajectory(
            traj_rl,
            save_path=save_path,
            show_angle_plot=True,
            fps=50
        )
        print("Animation saved!")
    else:
        print("Displaying animation window (close to continue)...")
        anim = animate_trajectory(
            traj_rl,
            show_angle_plot=True,
            fps=50
        )


def demo_comparison_animation(vec_env, model, planner, start_state, save_path=None):
    """
    Demonstrate side-by-side comparison animation (RL vs Classical).

    Args:
        vec_env: VecNormalize wrapped environment
        model: Trained SAC model
        planner: TrajectoryPlanner
        start_state: Initial state
        save_path: Optional save path
    """
    print("\n" + "="*80)
    print("DEMO 3: RL vs Classical Comparison Animation")
    print("="*80)
    print("\nThis creates a 2x2 comparison showing:")
    print("  - Top row: RL animation (left), Classical animation (right)")
    print("  - Bottom row: Angle comparison plot")
    print("\nRunning both controllers from same initial state...")

    # Collect both trajectories
    traj_rl, _ = rollout_rl_timed(model, vec_env, start_state, max_seconds=10.0)
    traj_classical, _ = rollout_classical_timed(planner, vec_env, start_state, max_seconds=10.0)

    print(f"RL trajectory: {len(traj_rl)} steps")
    print(f"Classical trajectory: {len(traj_classical)} steps")

    # Create comparison animation
    print("\nCreating comparison animation...")
    if save_path:
        print(f"Saving to: {save_path}")
        anim = animate_comparison(
            traj_rl,
            traj_classical,
            save_path=save_path,
            fps=50
        )
        print("Comparison animation saved!")
    else:
        print("Displaying comparison (close to continue)...")
        anim = animate_comparison(
            traj_rl,
            traj_classical,
            fps=50
        )


def main(args):
    """Main demonstration script."""
    print("="*80)
    print("Cart-Pendulum Visualization Demo")
    print("="*80)

    # Resolve paths
    model_path = Path(args.model).resolve()
    vecnorm_path = Path(args.vecnorm).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train a model first with: python scripts/train.py"
        )

    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found: {vecnorm_path}\n"
            f"Please train a model first with: python scripts/train.py"
        )

    print(f"\nLoading model from: {model_path}")
    print(f"Loading normalization from: {vecnorm_path}")

    # Load model and environment
    def make_env():
        env = CartPendulumEnv()
        return TimeLimit(env, max_episode_steps=1000)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load(str(model_path), device='cpu')
    planner = TrajectoryPlanner(umax=10.0)

    # Choose an interesting initial state (challenging swing-up)
    start_state = np.array([
        np.deg2rad(-160),  # θ₀ = -160° (nearly inverted)
        0.0,               # θ̇₀ = 0
        0.0,               # x₀ = 0
        0.0                # ẋ₀ = 0
    ])

    print(f"\nInitial state: θ₀={np.rad2deg(start_state[0]):.1f}°")

    # Run demos based on flags
    if args.skip_pygame:
        print("\nSkipping pygame demo (--skip-pygame flag set)")
    else:
        try:
            demo_pygame_rendering(vec_env, model, start_state, max_seconds=10.0)
        except ImportError as e:
            print(f"\nWarning: Pygame not available, skipping real-time rendering demo")
            print(f"Install with: pip install gymnasium[classic-control]")
            print(f"Error: {e}")

    # Matplotlib animations
    if args.save_animations:
        # Create output directory
        output_dir = Path("runs/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Single trajectory animation
        demo_matplotlib_animation(
            vec_env, model, planner, start_state,
            save_path=str(output_dir / "rl_trajectory.mp4")
        )

        # Comparison animation
        demo_comparison_animation(
            vec_env, model, planner, start_state,
            save_path=str(output_dir / "rl_vs_classical.mp4")
        )

        print(f"\n✓ All animations saved to: {output_dir}/")
    else:
        # Show interactive windows
        demo_matplotlib_animation(vec_env, model, planner, start_state)
        demo_comparison_animation(vec_env, model, planner, start_state)

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)
    print("\nNext steps:")
    print("  - Try different initial states by editing start_state in main()")
    print("  - Save animations with: --save-animations")
    print("  - Analyze Basin of Attraction with: python scripts/analyze_boa.py")
    print("  - Read the docs in README.md for more visualization options")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate cart-pendulum visualization capabilities"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/sac_train/phase2/sac_model.zip",
        help="Path to trained SAC model (default: runs/sac_train/phase2/sac_model.zip)"
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default="runs/sac_train/phase2/vecnormalize.pkl",
        help="Path to VecNormalize stats (default: runs/sac_train/phase2/vecnormalize.pkl)"
    )
    parser.add_argument(
        "--save-animations",
        action="store_true",
        help="Save animations to runs/visualizations/ instead of displaying"
    )
    parser.add_argument(
        "--skip-pygame",
        action="store_true",
        help="Skip pygame rendering demo (useful if pygame not installed)"
    )

    args = parser.parse_args()
    main(args)
