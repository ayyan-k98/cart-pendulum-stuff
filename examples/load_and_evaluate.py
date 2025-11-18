#!/usr/bin/env python3
"""
Example: Loading and evaluating a trained checkpoint.

This script demonstrates how to load a previously trained model and run
evaluations programmatically (not just via CLI).

Author: Cart-Pendulum Research Team
License: MIT
"""

import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import TimeLimit

from src import (
    CartPendulumEnv,
    rollout_rl_timed,
    animate_trajectory,
)


def load_checkpoint_simple(model_path: str, vecnorm_path: str = None):
    """
    Simple function to load a checkpoint.

    Args:
        model_path: Path to .zip model file
        vecnorm_path: Path to .pkl vecnorm file (optional, will auto-detect)

    Returns:
        model: Loaded SAC model
        vec_env: VecNormalize environment
    """
    model_path = Path(model_path)

    # Auto-detect vecnorm path if not provided
    if vecnorm_path is None:
        vecnorm_path = model_path.parent / "vecnormalize.pkl"

    # Load model
    model = SAC.load(str(model_path), device='cpu')

    # Create environment with normalization
    def make_env():
        env = CartPendulumEnv()
        return TimeLimit(env, max_episode_steps=1000)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    return model, vec_env


def example_1_basic_evaluation():
    """Example 1: Load checkpoint and evaluate on one state."""
    print("\n" + "="*80)
    print("Example 1: Basic Evaluation")
    print("="*80 + "\n")

    # Load checkpoint
    model, vec_env = load_checkpoint_simple(
        "runs/sac_train/phase2/sac_model.zip"
    )
    print("✓ Checkpoint loaded\n")

    # Define initial state
    theta0_deg = -150
    state = np.array([np.deg2rad(theta0_deg), 0.0, 0.0, 0.0])

    print(f"Evaluating from θ₀={theta0_deg}°...")

    # Run rollout
    trajectory, timing = rollout_rl_timed(model, vec_env, state, max_seconds=10.0)

    # Check result
    final_theta = np.rad2deg(trajectory['theta'].iloc[-1])
    success = abs(trajectory['theta'].iloc[-1]) < np.deg2rad(10)

    print(f"  Final angle: {final_theta:+.2f}°")
    print(f"  Success: {'✓ YES' if success else '✗ NO'}")
    print(f"  Inference time: {timing['mean_inference_ms']:.3f}ms")


def example_2_multiple_states():
    """Example 2: Evaluate on multiple initial states."""
    print("\n" + "="*80)
    print("Example 2: Multiple Initial States")
    print("="*80 + "\n")

    model, vec_env = load_checkpoint_simple(
        "runs/sac_train/phase2/sac_model.zip"
    )
    print("✓ Checkpoint loaded\n")

    # Test different initial angles
    test_angles = [-180, -120, -60, 0, 60, 120, 180]

    print(f"Testing {len(test_angles)} different initial angles:\n")

    for theta0_deg in test_angles:
        state = np.array([np.deg2rad(theta0_deg), 0.0, 0.0, 0.0])
        trajectory, timing = rollout_rl_timed(model, vec_env, state, max_seconds=10.0)

        final_theta = np.rad2deg(trajectory['theta'].iloc[-1])
        success = abs(trajectory['theta'].iloc[-1]) < np.deg2rad(10)

        status = "✓" if success else "✗"
        print(f"  θ₀={theta0_deg:+4d}° → θ_f={final_theta:+6.2f}° {status}")


def example_3_save_animation():
    """Example 3: Load checkpoint and create animation."""
    print("\n" + "="*80)
    print("Example 3: Save Animation")
    print("="*80 + "\n")

    model, vec_env = load_checkpoint_simple(
        "runs/sac_train/phase2/sac_model.zip"
    )
    print("✓ Checkpoint loaded\n")

    # Choose interesting initial state
    state = np.array([np.deg2rad(-170), 0.0, 0.0, 0.0])

    print("Running rollout from θ₀=-170°...")
    trajectory, timing = rollout_rl_timed(model, vec_env, state, max_seconds=8.0)

    # Create output directory
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)

    # Save animation
    output_path = output_dir / "demo_trajectory.mp4"
    print(f"Creating animation: {output_path}")

    anim = animate_trajectory(
        trajectory,
        save_path=str(output_path),
        show_angle_plot=True,
        fps=50
    )

    print(f"✓ Animation saved to {output_path}")


def example_4_custom_environment():
    """Example 4: Evaluate on custom environment configuration."""
    print("\n" + "="*80)
    print("Example 4: Custom Environment Configuration")
    print("="*80 + "\n")

    # Load model
    model = SAC.load("runs/sac_train/phase2/sac_model.zip", device='cpu')

    # Create environment with CUSTOM PARAMETERS
    def make_custom_env():
        env = CartPendulumEnv(
            rk4_substeps=20,  # Higher accuracy
            c_theta=(0.01, 0.05),  # Angular friction randomization
            c_x=(0.01, 0.05),      # Linear friction randomization
            reward_weights={
                'theta': 2.0,  # Penalize angle more
                'x': 0.2,      # Care less about position
            }
        )
        return TimeLimit(env, max_episode_steps=1000)

    vec_env = DummyVecEnv([make_custom_env])

    # Load normalization (from training)
    vec_env = VecNormalize.load(
        "runs/sac_train/phase2/vecnormalize.pkl",
        vec_env
    )
    vec_env.training = False
    vec_env.norm_reward = False

    print("✓ Custom environment created with:")
    print("  - 20 RK4 substeps (higher accuracy)")
    print("  - Friction randomization")
    print("  - Custom reward weights\n")

    # Test
    state = np.array([np.deg2rad(-140), 0.0, 0.0, 0.0])
    trajectory, timing = rollout_rl_timed(model, vec_env, state, max_seconds=10.0)

    final_theta = np.rad2deg(trajectory['theta'].iloc[-1])
    success = abs(trajectory['theta'].iloc[-1]) < np.deg2rad(10)

    print(f"Result with custom environment:")
    print(f"  Final angle: {final_theta:+.2f}°")
    print(f"  Success: {'✓ YES' if success else '✗ NO'}")


def example_5_access_policy_directly():
    """Example 5: Access policy network directly (advanced)."""
    print("\n" + "="*80)
    print("Example 5: Direct Policy Access (Advanced)")
    print("="*80 + "\n")

    model, vec_env = load_checkpoint_simple(
        "runs/sac_train/phase2/sac_model.zip"
    )

    # Reset environment
    obs = vec_env.reset()

    # Get action from policy (deterministic)
    action, _ = model.predict(obs, deterministic=True)
    print(f"Observation: {obs[0]}")
    print(f"Action (deterministic): {action[0]:.4f} N\n")

    # Get action distribution (stochastic)
    action_stochastic, _ = model.predict(obs, deterministic=False)
    print(f"Action (stochastic): {action_stochastic[0]:.4f} N\n")

    # Access actor network directly
    import torch
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).to(model.device)
        mean_actions = model.actor.mu(obs_tensor)
        print(f"Actor network mean output: {mean_actions[0].cpu().numpy():.4f} N")
        print(f"  (This is the deterministic policy action)")

    # Access critic networks (Q-functions)
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).to(model.device)
        action_tensor = torch.FloatTensor(action).to(model.device)

        q1 = model.critic.q1_forward(obs_tensor, action_tensor)
        q2 = model.critic.q2_forward(obs_tensor, action_tensor)

        print(f"\nCritic Q-values:")
        print(f"  Q1(s,a) = {q1[0].cpu().numpy():.4f}")
        print(f"  Q2(s,a) = {q2[0].cpu().numpy():.4f}")
        print(f"  (These estimate expected return from current state)")


if __name__ == "__main__":
    # Check if checkpoint exists
    checkpoint_path = Path("runs/sac_train/phase2/sac_model.zip")

    if not checkpoint_path.exists():
        print("\n" + "="*80)
        print("ERROR: No checkpoint found!")
        print("="*80)
        print(f"\nExpected checkpoint at: {checkpoint_path}")
        print("\nPlease train a model first:")
        print("  python scripts/train.py")
        print("\nOr specify a different checkpoint path in the examples above.")
        print("="*80 + "\n")
        exit(1)

    # Run examples
    print("\n" + "="*80)
    print("Loading and Evaluating Checkpoints - Examples")
    print("="*80)

    example_1_basic_evaluation()
    example_2_multiple_states()
    example_3_save_animation()
    example_4_custom_environment()
    example_5_access_policy_directly()

    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80 + "\n")
