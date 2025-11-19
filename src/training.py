"""
Training pipeline for Cart-Pendulum RL agent using Soft Actor-Critic (SAC).

This module implements a two-phase curriculum learning approach:
    Phase 1: Stabilization - Learn to balance from near-upright states
    Phase 2: Swing-up - Learn full swing-up from arbitrary angles

The curriculum makes learning more efficient by first mastering the easier
stabilization task before tackling the full swing-up problem.

Features:
- Two-phase curriculum learning
- Domain randomization via friction ranges
- Observation normalization with VecNormalize
- Parallel environment execution
- Model checkpointing and resume capability
- Progress callbacks
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any
from functools import partial

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

from .environment import CartPendulumEnv
from .utils import check_device_available, ensure_directory_exists


class TextProgressCallback(BaseCallback):
    """
    Simple callback that prints training progress at regular intervals.

    This is more suitable for notebook/script usage than the default
    rich progress bar which can clutter outputs.
    """

    def __init__(
        self,
        total_timesteps: int,
        update_every: int = 10_000,
        name: str = "train"
    ):
        """
        Initialize progress callback.

        Args:
            total_timesteps: Total training timesteps
            update_every: Print update every N timesteps
            name: Name prefix for progress messages
        """
        super().__init__()
        self.total = int(total_timesteps)
        self.update_every = int(update_every)
        self.name = name
        self._last_n = 0

    def _on_step(self) -> bool:
        """Called after each environment step."""
        n = self.num_timesteps
        if n - self._last_n >= self.update_every or n >= self.total:
            pct = 100.0 * n / max(1, self.total)
            print(f"[{self.name}] Timesteps: {n}/{self.total} ({pct:5.1f}%)", flush=True)
            self._last_n = n
        return True


def train_sac(
    total_steps: int = 500_000,
    n_envs: int = 8,
    train_substeps: int = 6,
    batch_size: int = 768,
    gradient_steps: int = 3,
    seed: int = 42,
    device: str = 'auto',
    out_dir: str = 'runs/sac_train',
    soft_wall_k: float = 0.5,
    du_weight: float = 1e-3,
    two_phase: bool = True,
    train_with_friction: bool = True,
    verbose: bool = True
) -> Tuple[str, str]:
    """
    Train a SAC agent for cart-pendulum control.

    This function implements a complete training pipeline with optional
    two-phase curriculum learning for better sample efficiency.

    Args:
        total_steps: Total training timesteps for main (phase 2) training
        n_envs: Number of parallel environments
        train_substeps: RK4 integration substeps for training
        batch_size: SAC batch size
        gradient_steps: Number of gradient steps per environment step
        seed: Random seed for reproducibility
        device: 'cuda', 'cpu', or 'auto'
        out_dir: Output directory for models and logs
        soft_wall_k: Soft wall penalty coefficient
        du_weight: Action smoothness penalty weight
        two_phase: If True, run phase 1 (stabilization) before phase 2
        train_with_friction: If True, use friction randomization during training
            - True: c_theta ∈ [0.0, 0.03], c_x ∈ [0.0, 0.05] (robust to friction)
            - False: c_theta = 0.0, c_x = 0.0 (frictionless training)
            - Allows comparison of models trained with/without friction modeling
        verbose: Print progress messages

    Returns:
        Tuple of (model_path, vecnorm_path) for the trained model

    Raises:
        ValueError: If invalid parameters provided
        RuntimeError: If training fails

    Example:
        >>> # Train with friction (default)
        >>> model_path, vecnorm_path = train_sac(
        ...     total_steps=500_000,
        ...     n_envs=8,
        ...     device='cuda',
        ...     train_with_friction=True
        ... )
        >>> # Train without friction
        >>> model_path, vecnorm_path = train_sac(
        ...     total_steps=500_000,
        ...     n_envs=8,
        ...     device='cuda',
        ...     train_with_friction=False
        ... )
    """
    # Validate inputs
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if n_envs <= 0:
        raise ValueError(f"n_envs must be positive, got {n_envs}")

    # Check device availability
    device = check_device_available(device)
    if verbose:
        print(f"Using device: {device}")

    # Create output directory
    ensure_directory_exists(out_dir)

    # Configure multiprocessing for parallel environments
    start_method = {}
    if os.name == "posix":
        # Use spawn for notebook compatibility, fork for scripts
        method = "spawn" if "ipykernel" in sys.modules else "fork"
        start_method = {"start_method": method}

    # Configure friction based on train_with_friction flag
    # Phase 1: Light friction randomization
    c_theta_p1 = (0.0, 0.03) if train_with_friction else 0.0
    c_x_p1 = (0.0, 0.05) if train_with_friction else 0.0
    # Phase 2: Heavier friction randomization
    c_theta_p2 = (0.0, 0.05) if train_with_friction else 0.0
    c_x_p2 = (0.0, 0.08) if train_with_friction else 0.0

    # Phase 1: Stabilization (optional but recommended)
    if two_phase:
        if verbose:
            print("\n" + "="*80)
            print("PHASE 1: Stabilization Training")
            print("="*80)

        p1_out_dir = os.path.join(out_dir, "phase1")
        ensure_directory_exists(p1_out_dir)

        # Phase 1 environment: start near upright
        def make_p1_env():
            env = CartPendulumEnv(
                curriculum_phase="stabilization",
                rk4_substeps=train_substeps,
                c_theta=c_theta_p1,  # Friction randomization (or zero)
                c_x=c_x_p1,
                soft_wall_k=soft_wall_k,
                du_weight=du_weight
            )
            return TimeLimit(env, max_episode_steps=1000)

        env_p1 = make_vec_env(
            make_p1_env,
            n_envs=n_envs,
            seed=seed,
            vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
            vec_env_kwargs=start_method
        )
        env_p1 = VecNormalize(env_p1, norm_obs=True, norm_reward=True)

        # Phase 1 SAC parameters
        p1_steps = min(200_000, total_steps // 3)  # ~1/3 of total budget
        sac_p1 = SAC(
            policy="MlpPolicy",
            env=env_p1,
            learning_rate=3e-4,
            buffer_size=int(1e6),
            batch_size=1024,
            gamma=0.99,
            tau=0.005,
            train_freq=(64, "step"),
            gradient_steps=64,
            learning_starts=10_000,
            ent_coef='auto_0.1',
            policy_kwargs=dict(net_arch=[256, 256], use_sde=False),
            verbose=0,
            device=device,
            seed=seed
        )

        if verbose:
            print(f"Training Phase 1 for {p1_steps:,} steps...")
            print(f"TensorBoard logs: {p1_out_dir}/logs/")

        # Configure TensorBoard logging
        tb_log_path = os.path.join(p1_out_dir, "logs")
        ensure_directory_exists(tb_log_path)
        sac_p1.set_logger(configure(tb_log_path, ["tensorboard"]))

        callback_p1 = TextProgressCallback(p1_steps, update_every=10_000, name="Phase1")
        sac_p1.learn(total_timesteps=p1_steps, callback=callback_p1, progress_bar=False)

        # Save Phase 1 model
        p1_model_path = os.path.join(p1_out_dir, "sac_stabilize_model.zip")
        p1_vecnorm_path = os.path.join(p1_out_dir, "vecnormalize_stabilize.pkl")
        sac_p1.save(p1_model_path)
        env_p1.save(p1_vecnorm_path)

        if verbose:
            print(f"Phase 1 complete. Model saved to {p1_model_path}")

        # Clean up
        env_p1.close()

    # Phase 2: Swing-up (main training)
    if verbose:
        print("\n" + "="*80)
        print("PHASE 2: Swing-Up Training")
        print("="*80)

    p2_out_dir = os.path.join(out_dir, "phase2")
    ensure_directory_exists(p2_out_dir)

    # Phase 2 environment: full swing-up with heavier friction randomization
    def make_p2_env():
        env = CartPendulumEnv(
            curriculum_phase="swingup",
            rk4_substeps=train_substeps,
            c_theta=c_theta_p2,  # Heavier friction randomization (or zero)
            c_x=c_x_p2,
            soft_wall_k=soft_wall_k,
            du_weight=du_weight
        )
        return TimeLimit(env, max_episode_steps=2000)

    env_p2 = make_vec_env(
        make_p2_env,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        vec_env_kwargs=start_method
    )
    env_p2 = VecNormalize(env_p2, norm_obs=True, norm_reward=False)

    # Phase 2 SAC parameters
    sac_params = dict(
        learning_rate=3e-4,
        buffer_size=int(2e6),
        batch_size=batch_size,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=gradient_steps,
        ent_coef='auto_0.1',
        policy_kwargs=dict(net_arch=[256, 256], use_sde=False),
        verbose=0,
        device=device,
        seed=seed
    )

    if two_phase:
        # Load Phase 1 model for warm start
        if verbose:
            print("Loading Phase 1 model for warm start...")

        try:
            sac_p2 = SAC.load(
                p1_model_path,
                env=env_p2,
                **sac_params,
                learning_starts=0  # Already pre-trained
            )
            # Load Phase 1 normalization stats
            env_p2 = VecNormalize.load(p1_vecnorm_path, env_p2)
            env_p2.training = True
            env_p2.norm_reward = False  # Don't normalize reward for phase 2
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load Phase 1 model: {e}")
                print("Starting Phase 2 from scratch...")
            sac_p2 = SAC("MlpPolicy", env_p2, **sac_params, learning_starts=10_000)
    else:
        # Train from scratch
        sac_p2 = SAC("MlpPolicy", env_p2, **sac_params, learning_starts=10_000)

    if verbose:
        print(f"Training Phase 2 for {total_steps:,} steps...")
        print(f"TensorBoard logs: {p2_out_dir}/logs/")

    # Configure TensorBoard logging
    tb_log_path = os.path.join(p2_out_dir, "logs")
    ensure_directory_exists(tb_log_path)
    sac_p2.set_logger(configure(tb_log_path, ["tensorboard"]))

    callback_p2 = TextProgressCallback(total_steps, update_every=20_000, name="Phase2")

    try:
        sac_p2.learn(
            total_timesteps=total_steps,
            callback=callback_p2,
            progress_bar=False,
            reset_num_timesteps=(not two_phase)  # Continue counting if warm start
        )
    except KeyboardInterrupt:
        if verbose:
            print("\nTraining interrupted by user.")
    except Exception as e:
        env_p2.close()
        raise RuntimeError(f"Training failed: {e}")

    # Save final model
    model_path = os.path.join(p2_out_dir, "sac_model.zip")
    vecnorm_path = os.path.join(p2_out_dir, "vecnormalize.pkl")

    sac_p2.save(model_path)
    env_p2.save(vecnorm_path)

    if verbose:
        print(f"\nTraining complete!")
        print(f"Model saved to: {model_path}")
        print(f"VecNormalize saved to: {vecnorm_path}")

    env_p2.close()

    return model_path, vecnorm_path


def finetune_sac(
    model_path: str,
    vecnorm_path: str,
    total_steps: int = 500_000,
    n_envs: int = 8,
    train_substeps: int = 6,
    batch_size: int = 1024,
    gradient_steps: int = 64,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = 'auto',
    out_dir: str = 'runs/sac_finetune',
    soft_wall_k: float = 0.6,
    du_weight: float = 2e-3,
    verbose: bool = True
) -> Tuple[str, str]:
    """
    Fine-tune a pre-trained SAC model with adjusted parameters.

    Useful for:
    - Improving performance with different hyperparameters
    - Adapting to different friction/dynamics
    - Increasing robustness

    Args:
        model_path: Path to pre-trained SAC model (.zip)
        vecnorm_path: Path to VecNormalize stats (.pkl)
        total_steps: Additional training timesteps
        n_envs: Number of parallel environments
        train_substeps: RK4 substeps
        batch_size: SAC batch size (often larger for finetuning)
        gradient_steps: Gradient steps per env step (often larger)
        learning_rate: Learning rate (often lower for finetuning)
        seed: Random seed
        device: 'cuda', 'cpu', or 'auto'
        out_dir: Output directory
        soft_wall_k: Soft wall penalty (can increase for robustness)
        du_weight: Action smoothness weight (can increase for smoother policies)
        verbose: Print progress

    Returns:
        Tuple of (finetuned_model_path, vecnorm_path)

    Example:
        >>> ft_model, ft_vecnorm = finetune_sac(
        ...     model_path='runs/sac_train/phase2/sac_model.zip',
        ...     vecnorm_path='runs/sac_train/phase2/vecnormalize.pkl',
        ...     total_steps=500_000,
        ...     learning_rate=1e-4
        ... )
    """
    if verbose:
        print("\n" + "="*80)
        print("FINE-TUNING")
        print("="*80)
        print(f"Loading model from: {model_path}")
        print(f"Loading VecNormalize from: {vecnorm_path}")

    # Check files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize not found: {vecnorm_path}")

    device = check_device_available(device)
    ensure_directory_exists(out_dir)

    # Setup environment
    start_method = {}
    if os.name == "posix":
        method = "spawn" if "ipykernel" in sys.modules else "fork"
        start_method = {"start_method": method}

    def make_env():
        env = CartPendulumEnv(
            curriculum_phase="swingup",
            rk4_substeps=train_substeps,
            c_theta=(0.0, 0.05),
            c_x=(0.0, 0.08),
            soft_wall_k=soft_wall_k,
            du_weight=du_weight
        )
        return TimeLimit(env, max_episode_steps=2000)

    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        vec_env_kwargs=start_method
    )

    # Load VecNormalize (with existing statistics)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = True
    env.norm_reward = False

    # Load model
    model = SAC.load(
        model_path,
        env=env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_steps=gradient_steps,
        learning_starts=0,  # No warm-up needed
        device=device
    )

    if verbose:
        print(f"Fine-tuning for {total_steps:,} additional steps...")
        print(f"TensorBoard logs: {out_dir}/logs/")

    # Configure TensorBoard logging
    tb_log_path = os.path.join(out_dir, "logs")
    ensure_directory_exists(tb_log_path)
    model.set_logger(configure(tb_log_path, ["tensorboard"]))

    callback = TextProgressCallback(total_steps, update_every=20_000, name="Finetune")

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=False  # Continue from existing count
        )
    except KeyboardInterrupt:
        if verbose:
            print("\nFine-tuning interrupted by user.")
    except Exception as e:
        env.close()
        raise RuntimeError(f"Fine-tuning failed: {e}")

    # Save
    ft_model_path = os.path.join(out_dir, "sac_model.zip")
    ft_vecnorm_path = os.path.join(out_dir, "vecnormalize.pkl")

    model.save(ft_model_path)
    env.save(ft_vecnorm_path)

    if verbose:
        print(f"\nFine-tuning complete!")
        print(f"Model saved to: {ft_model_path}")
        print(f"VecNormalize saved to: {ft_vecnorm_path}")

    env.close()

    return ft_model_path, ft_vecnorm_path
