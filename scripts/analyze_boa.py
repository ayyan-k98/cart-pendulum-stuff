#!/usr/bin/env python3
"""
Basin of Attraction Analysis for Cart-Pendulum Controllers.

This script evaluates both RL and classical controllers across a grid of initial
states to map out their basins of attraction and compare performance metrics.

Usage:
    # Full analysis with default grid (41x31 = 1271 states)
    python scripts/analyze_boa.py \\
        --model runs/sac_train/phase2/sac_model.zip \\
        --vecnorm runs/sac_train/phase2/vecnormalize.pkl

    # Quick test with smaller grid
    python scripts/analyze_boa.py \\
        --model runs/sac_train/phase2/sac_model.zip \\
        --vecnorm runs/sac_train/phase2/vecnormalize.pkl \\
        --n-theta 21 --n-theta-dot 21
"""

import argparse
import sys
import os
import math

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analysis import create_state_grid, evaluate_state_grid, summarize_results
from src.visualization import (
    plot_basin_of_attraction,
    plot_timing_comparison,
    plot_success_comparison,
    plot_metric_histogram
)
from src.utils import ensure_directory_exists


def parse_args():
    parser = argparse.ArgumentParser(
        description='Perform Basin of Attraction analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained SAC model (.zip)')
    parser.add_argument('--vecnorm', type=str, required=True,
                       help='Path to VecNormalize stats (.pkl)')

    # Grid parameters
    parser.add_argument('--n-theta', type=int, default=41,
                       help='Number of theta grid points')
    parser.add_argument('--n-theta-dot', type=int, default=31,
                       help='Number of theta_dot grid points')

    # Evaluation parameters
    parser.add_argument('--c-theta', type=float, default=0.02,
                       help='Angular friction coefficient')
    parser.add_argument('--c-x', type=float, default=0.05,
                       help='Linear friction coefficient')
    parser.add_argument('--eval-substeps', type=int, default=10,
                       help='RK4 substeps for evaluation')
    parser.add_argument('--max-seconds', type=float, default=20.0,
                       help='Maximum episode duration (seconds)')

    # Output
    parser.add_argument('--out-dir', type=str, default='runs/boa_analysis',
                       help='Output directory for results and plots')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("BASIN OF ATTRACTION ANALYSIS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"VecNormalize: {args.vecnorm}")
    print(f"Grid size: {args.n_theta} × {args.n_theta_dot} = {args.n_theta * args.n_theta_dot} states")
    print(f"Total evaluations: {2 * args.n_theta * args.n_theta_dot} (both controllers)")
    print(f"Output directory: {args.out_dir}")
    print("=" * 80)

    # Create output directory
    ensure_directory_exists(args.out_dir)

    # 1. Create state grid
    print("\n1. Creating state grid...")
    state_grid = create_state_grid(
        theta_range=(-math.pi, math.pi),
        theta_dot_range=(-3.0, 3.0),
        n_theta=args.n_theta,
        n_theta_dot=args.n_theta_dot
    )
    print(f"   Created {len(state_grid)} states")

    # 2. Evaluate grid
    print("\n2. Evaluating grid...")
    results_df = evaluate_state_grid(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        state_grid=state_grid,
        max_seconds=args.max_seconds,
        c_theta=args.c_theta,
        c_x=args.c_x,
        eval_substeps=args.eval_substeps,
        verbose=True
    )

    # 3. Save results
    results_path = os.path.join(args.out_dir, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n3. Results saved to: {results_path}")

    # 4. Generate summary statistics
    print("\n4. Computing summary statistics...")
    summary = summarize_results(results_df, verbose=True)

    # 5. Generate visualizations
    print("\n5. Generating visualizations...")

    # Basin of Attraction plots
    print("   - Basin of Attraction (RL)...")
    plot_basin_of_attraction(
        results_df, 'success', 'rl',
        save_path=os.path.join(args.out_dir, 'boa_rl.png')
    )

    print("   - Basin of Attraction (Classical)...")
    plot_basin_of_attraction(
        results_df, 'success', 'classical',
        save_path=os.path.join(args.out_dir, 'boa_classical.png')
    )

    # Settling time plots
    print("   - Settling Time (RL)...")
    plot_basin_of_attraction(
        results_df, 'settling_time', 'rl',
        save_path=os.path.join(args.out_dir, 'settling_rl.png')
    )

    print("   - Settling Time (Classical)...")
    plot_basin_of_attraction(
        results_df, 'settling_time', 'classical',
        save_path=os.path.join(args.out_dir, 'settling_classical.png')
    )

    # THE MONEY PLOT: Timing comparison
    print("   - Timing Comparison (THE MONEY PLOT)...")
    plot_timing_comparison(
        results_df,
        save_path=os.path.join(args.out_dir, 'timing_comparison.png')
    )

    # Side-by-side success comparison
    print("   - Success Comparison...")
    plot_success_comparison(
        results_df,
        save_path=os.path.join(args.out_dir, 'success_comparison.png')
    )

    # Histograms
    print("   - Control Effort Histogram...")
    plot_metric_histogram(
        results_df, 'control_effort',
        save_path=os.path.join(args.out_dir, 'histogram_control_effort.png')
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"All results saved to: {args.out_dir}/")
    print("\nKey files:")
    print(f"  - results.csv: Full numerical results")
    print(f"  - boa_*.png: Basin of Attraction plots")
    print(f"  - timing_comparison.png: THE MONEY PLOT")
    print(f"  - success_comparison.png: Side-by-side BoA comparison")
    print("=" * 80)


if __name__ == '__main__':
    main()
