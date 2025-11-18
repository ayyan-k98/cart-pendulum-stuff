#!/usr/bin/env python3
"""
Main training script for Cart-Pendulum SAC agent.

This script provides a command-line interface for training the SAC agent
with configurable hyperparameters. It can be run directly or imported.

Usage:
    # Quick test (reduced timesteps)
    python scripts/train.py --total-steps 100000 --n-envs 4

    # Full training
    python scripts/train.py --total-steps 1500000 --n-envs 8 --device cuda

    # Single-phase training (no curriculum)
    python scripts/train.py --total-steps 500000 --no-two-phase

Example:
    >>> # From Python
    >>> from scripts.train import main
    >>> main(['--total-steps', '100000', '--device', 'cpu'])
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import train_sac, finetune_sac


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SAC agent for cart-pendulum control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument(
        '--total-steps', type=int, default=500_000,
        help='Total training timesteps for Phase 2 (swing-up)'
    )
    parser.add_argument(
        '--n-envs', type=int, default=8,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--train-substeps', type=int, default=6,
        help='RK4 integration substeps for training'
    )
    parser.add_argument(
        '--batch-size', type=int, default=768,
        help='SAC batch size'
    )
    parser.add_argument(
        '--gradient-steps', type=int, default=3,
        help='Gradient steps per environment step'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'],
        help='Device to use for training'
    )

    # Output
    parser.add_argument(
        '--out-dir', type=str, default='runs/sac_train',
        help='Output directory for models and logs'
    )

    # Environment parameters
    parser.add_argument(
        '--soft-wall-k', type=float, default=0.5,
        help='Soft wall penalty coefficient'
    )
    parser.add_argument(
        '--du-weight', type=float, default=1e-3,
        help='Action smoothness penalty weight'
    )

    # Curriculum
    parser.add_argument(
        '--no-two-phase', action='store_true',
        help='Disable two-phase curriculum learning'
    )

    # Fine-tuning
    parser.add_argument(
        '--finetune', action='store_true',
        help='Fine-tune existing model instead of training from scratch'
    )
    parser.add_argument(
        '--model-path', type=str,
        help='Path to model for fine-tuning'
    )
    parser.add_argument(
        '--vecnorm-path', type=str,
        help='Path to VecNormalize for fine-tuning'
    )
    parser.add_argument(
        '--finetune-lr', type=float, default=1e-4,
        help='Learning rate for fine-tuning'
    )
    parser.add_argument(
        '--finetune-batch-size', type=int, default=1024,
        help='Batch size for fine-tuning'
    )
    parser.add_argument(
        '--finetune-gradient-steps', type=int, default=64,
        help='Gradient steps for fine-tuning'
    )

    # Verbosity
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress messages'
    )

    return parser.parse_args(args)


def main(args=None):
    """
    Main training function.

    Args:
        args: Command-line arguments (for testing). If None, uses sys.argv.
    """
    args = parse_args(args)

    print("=" * 80)
    print("Cart-Pendulum SAC Training")
    print("=" * 80)
    print(f"Total steps: {args.total_steps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.out_dir}")
    print(f"Two-phase curriculum: {not args.no_two_phase}")
    print("=" * 80)

    if args.finetune:
        # Fine-tuning mode
        if not args.model_path or not args.vecnorm_path:
            print("ERROR: --model-path and --vecnorm-path required for fine-tuning")
            sys.exit(1)

        model_path, vecnorm_path = finetune_sac(
            model_path=args.model_path,
            vecnorm_path=args.vecnorm_path,
            total_steps=args.total_steps,
            n_envs=args.n_envs,
            train_substeps=args.train_substeps,
            batch_size=args.finetune_batch_size,
            gradient_steps=args.finetune_gradient_steps,
            learning_rate=args.finetune_lr,
            seed=args.seed,
            device=args.device,
            out_dir=args.out_dir,
            soft_wall_k=args.soft_wall_k,
            du_weight=args.du_weight,
            verbose=not args.quiet
        )
    else:
        # Training mode
        model_path, vecnorm_path = train_sac(
            total_steps=args.total_steps,
            n_envs=args.n_envs,
            train_substeps=args.train_substeps,
            batch_size=args.batch_size,
            gradient_steps=args.gradient_steps,
            seed=args.seed,
            device=args.device,
            out_dir=args.out_dir,
            soft_wall_k=args.soft_wall_k,
            du_weight=args.du_weight,
            two_phase=not args.no_two_phase,
            verbose=not args.quiet
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print("=" * 80)

    return model_path, vecnorm_path


if __name__ == '__main__':
    main()
