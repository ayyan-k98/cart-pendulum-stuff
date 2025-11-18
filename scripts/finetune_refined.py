#!/usr/bin/env python3
"""
Fine-tune SAC model with refined reward weights for better performance.

This script fine-tunes an existing model with adjusted reward weights to:
1. Reduce jittery control (increase du_weight)
2. Fix cart centering (increase x and x_dot penalties)

Usage:
    python scripts/finetune_refined.py \\
        --model runs/sac_train/phase2/sac_model.zip \\
        --vecnorm runs/sac_train/phase2/vecnormalize.pkl \\
        --steps 500000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Note: Full implementation requires modifying train_sac to accept reward_weights
# For now, this demonstrates the intended usage

print("=" * 80)
print("REFINED REWARD TUNING")
print("=" * 80)
print("\nRecommended reward weight adjustments:")
print("  - x: 0.15 → 0.25 (stronger cart centering)")
print("  - x_dot: 0.01 → 0.02 (penalize cart velocity)")
print("  - du_weight: 1e-3 → 5e-3 (smoother control)")
print("\nNote: Modify CartPendulumEnv initialization in training script")
print("=" * 80)

# TODO: After modifying train_sac to accept reward_weights parameter:
# from src import finetune_sac
#
# refined_weights = {
#     'x': 0.25,       # Increased from 0.5 (default is 0.5 but was 0.15)
#     'x_dot': 0.02,   # Increased from 0.01
# }
#
# model_path, vecnorm_path = finetune_sac(
#     model_path=args.model,
#     vecnorm_path=args.vecnorm,
#     total_steps=args.steps,
#     du_weight=5e-3,  # Increased from 1e-3
#     # reward_weights=refined_weights  # TODO: Add this parameter
# )
