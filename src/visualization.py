"""
Visualization tools for Basin of Attraction and performance analysis.

This module provides publication-quality plotting functions for:
- Basin of Attraction heatmaps
- Settling time heatmaps
- Timing comparison plots (THE MONEY PLOT)
- Performance metric comparisons

Author: Cart-Pendulum Research Team
License: MIT
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


def plot_basin_of_attraction(
    results_df: pd.DataFrame,
    metric: str = 'success',
    controller: str = 'rl',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> Figure:
    """
    Create 2D heatmap of Basin of Attraction (θ₀ vs θ̇₀).

    Args:
        results_df: DataFrame from evaluate_state_grid with columns:
            ['theta_0', 'theta_dot_0', 'controller', 'success', 'settling_time', ...]
        metric: Metric to plot - options:
            - 'success': Binary success/failure (default)
            - 'settling_time': Time to settle (seconds)
            - 'control_effort': Total control effort
            - 'final_angle_error_deg': Final angle error (degrees)
            - 'initial_plan_time_ms': Planning time (classical only)
        controller: 'rl' or 'classical'
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        title: Optional custom title

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_basin_of_attraction(results_df, 'success', 'rl')
        >>> plt.show()
    """
    # Filter for specific controller
    data = results_df[results_df['controller'] == controller].copy()

    # Convert theta_0 to degrees for plotting
    data['theta_0_plot'] = np.rad2deg(data['theta_0'])

    # Pivot to grid format
    try:
        grid = data.pivot(
            index='theta_dot_0',
            columns='theta_0_plot',
            values=metric
        )
    except Exception as e:
        raise ValueError(f"Error creating pivot table: {e}. "
                        f"Make sure data has no duplicate (theta_0, theta_dot_0) pairs.")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colormap and labels based on metric
    if metric == 'success':
        # Binary colormap: Red=Fail, Green=Success
        cmap = mcolors.ListedColormap(['#d62728', '#2ca02c'])  # Red, Green
        cbar_label = 'Success'
        vmin, vmax = 0, 1
        if title is None:
            title = f'{controller.upper()}: Basin of Attraction'
    elif metric == 'settling_time':
        cmap = 'viridis_r'  # Reversed: darker = faster
        cbar_label = 'Settling Time (s)'
        vmin, vmax = None, None
        if title is None:
            title = f'{controller.upper()}: Settling Time'
    elif metric == 'control_effort':
        cmap = 'plasma'
        cbar_label = 'Control Effort (N·s)'
        vmin, vmax = None, None
        if title is None:
            title = f'{controller.upper()}: Control Effort'
    elif metric == 'initial_plan_time_ms':
        cmap = 'hot'
        cbar_label = 'Planning Time (ms)'
        vmin, vmax = None, None
        if title is None:
            title = f'{controller.upper()}: Initial Planning Time'
    else:
        cmap = 'viridis'
        cbar_label = metric.replace('_', ' ').title()
        vmin, vmax = None, None
        if title is None:
            title = f'{controller.upper()}: {cbar_label}'

    # Get extent for imshow
    theta_min = data['theta_0_plot'].min()
    theta_max = data['theta_0_plot'].max()
    theta_dot_min = data['theta_dot_0'].min()
    theta_dot_max = data['theta_dot_0'].max()

    # Plot heatmap
    im = ax.imshow(
        grid,
        cmap=cmap,
        aspect='auto',
        extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)

    # Labels and title
    ax.set_xlabel('Initial Angle θ₀ (degrees)', fontsize=14)
    ax.set_ylabel('Initial Angular Velocity θ̇₀ (rad/s)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_timing_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 6)
) -> Figure:
    """
    Create THE MONEY PLOT: 3-panel timing comparison.

    Panel 1: RL inference time (should be flat across state space)
    Panel 2: FFFB initial planning time (varies with state difficulty)
    Panel 3: FFFB per-step action time (should be relatively flat)

    This visualization clearly shows the computational trade-off:
    - RL: constant inference time, no planning
    - Classical: variable planning time (can be 100+ ms), fast action evaluation

    Args:
        results_df: DataFrame from evaluate_state_grid
        save_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_timing_comparison(results_df)
        >>> plt.show()
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Computational Cost Comparison: RL vs Classical', fontsize=18, fontweight='bold')

    # --- Panel 1: RL Inference Time ---
    rl_data = results_df[results_df['controller'] == 'rl'].copy()
    rl_data['theta_0_deg'] = np.rad2deg(rl_data['theta_0'])

    grid_rl = rl_data.pivot(
        index='theta_dot_0',
        columns='theta_0_deg',
        values='inference_time_mean_ms'
    )

    theta_min = rl_data['theta_0_deg'].min()
    theta_max = rl_data['theta_0_deg'].max()
    theta_dot_min = rl_data['theta_dot_0'].min()
    theta_dot_max = rl_data['theta_dot_0'].max()

    im1 = axes[0].imshow(
        grid_rl,
        cmap='viridis',
        aspect='auto',
        extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
        origin='lower',
        interpolation='nearest'
    )

    axes[0].set_title('RL: Inference Time\n(Constant Across States)', fontsize=14)
    axes[0].set_xlabel('θ₀ (deg)', fontsize=12)
    axes[0].set_ylabel('θ̇₀ (rad/s)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Time (ms)', fontsize=11)

    # --- Panel 2: FFFB Initial Planning Time (THE MONEY SHOT!) ---
    classical_data = results_df[results_df['controller'] == 'classical'].copy()
    classical_data['theta_0_deg'] = np.rad2deg(classical_data['theta_0'])

    grid_plan = classical_data.pivot(
        index='theta_dot_0',
        columns='theta_0_deg',
        values='initial_plan_time_ms'
    )

    im2 = axes[1].imshow(
        grid_plan,
        cmap='hot',
        aspect='auto',
        extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
        origin='lower',
        interpolation='nearest'
    )

    axes[1].set_title('Classical: Planning Time\n(Varies with State Difficulty)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('θ₀ (deg)', fontsize=12)
    axes[1].set_ylabel('θ̇₀ (rad/s)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Time (ms)', fontsize=11)

    # --- Panel 3: FFFB Per-Step Action Time ---
    grid_action = classical_data.pivot(
        index='theta_dot_0',
        columns='theta_0_deg',
        values='action_time_mean_ms'
    )

    im3 = axes[2].imshow(
        grid_action,
        cmap='viridis',
        aspect='auto',
        extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
        origin='lower',
        interpolation='nearest'
    )

    axes[2].set_title('Classical: Per-Step Time\n(Fast After Planning)', fontsize=14)
    axes[2].set_xlabel('θ₀ (deg)', fontsize=12)
    axes[2].set_ylabel('θ̇₀ (rad/s)', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('Time (ms)', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timing comparison to: {save_path}")

    return fig


def plot_success_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6)
) -> Figure:
    """
    Create side-by-side Basin of Attraction comparison for RL vs Classical.

    Args:
        results_df: DataFrame from evaluate_state_grid
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Basin of Attraction: RL vs Classical', fontsize=18, fontweight='bold')

    # Shared colormap for consistency
    cmap = mcolors.ListedColormap(['#d62728', '#2ca02c'])  # Red, Green

    for idx, controller in enumerate(['rl', 'classical']):
        data = results_df[results_df['controller'] == controller].copy()
        data['theta_0_deg'] = np.rad2deg(data['theta_0'])

        grid = data.pivot(
            index='theta_dot_0',
            columns='theta_0_deg',
            values='success'
        )

        theta_min = data['theta_0_deg'].min()
        theta_max = data['theta_0_deg'].max()
        theta_dot_min = data['theta_dot_0'].min()
        theta_dot_max = data['theta_dot_0'].max()

        im = axes[idx].imshow(
            grid,
            cmap=cmap,
            aspect='auto',
            extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
            origin='lower',
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )

        axes[idx].set_title(f'{controller.upper()}', fontsize=16)
        axes[idx].set_xlabel('θ₀ (deg)', fontsize=13)
        axes[idx].set_ylabel('θ̇₀ (rad/s)', fontsize=13)
        axes[idx].grid(True, alpha=0.3, linestyle='--')

        # Success rate annotation
        success_rate = data['success'].mean()
        axes[idx].text(
            0.02, 0.98,
            f'Success: {success_rate*100:.1f}%',
            transform=axes[idx].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    # Shared colorbar
    fig.colorbar(im, ax=axes, label='Success (0=Fail, 1=Success)', fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved success comparison to: {save_path}")

    return fig


def plot_metric_histogram(
    results_df: pd.DataFrame,
    metric: str = 'settling_time',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot histogram comparison of a metric for RL vs Classical.

    Args:
        results_df: DataFrame from evaluate_state_grid
        metric: Metric to plot ('settling_time', 'control_effort', etc.)
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    rl_data = results_df[results_df['controller'] == 'rl'][metric].dropna()
    classical_data = results_df[results_df['controller'] == 'classical'][metric].dropna()

    # Plot histograms
    ax.hist(rl_data, bins=30, alpha=0.5, label='RL', color='#1f77b4', edgecolor='black')
    ax.hist(classical_data, bins=30, alpha=0.5, label='Classical', color='#ff7f0e', edgecolor='black')

    # Add mean lines
    ax.axvline(rl_data.mean(), color='#1f77b4', linestyle='--', linewidth=2, label=f'RL Mean: {rl_data.mean():.2f}')
    ax.axvline(classical_data.mean(), color='#ff7f0e', linestyle='--', linewidth=2, label=f'Classical Mean: {classical_data.mean():.2f}')

    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'{metric.replace("_", " ").title()} Distribution: RL vs Classical', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to: {save_path}")

    return fig
