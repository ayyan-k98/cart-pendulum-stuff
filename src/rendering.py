"""
Rendering and animation utilities for cart-pendulum visualizations.

This module provides matplotlib-based animations for post-hoc analysis.
For real-time rendering, use CartPendulumEnv with render_mode='human'.

Features:
- Animate trajectory from DataFrame
- Side-by-side animation + angle plot
- Save to MP4/GIF for papers/presentations
- No pygame dependency (uses matplotlib only)
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import pandas as pd


def animate_trajectory(
    trajectory_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    show_angle_plot: bool = True,
    fps: int = 50
):
    """
    Create matplotlib animation from trajectory DataFrame.

    This function creates a publication-quality animation showing:
    - Left panel: Cart-pendulum animation
    - Right panel: Angle vs time plot (optional)

    Args:
        trajectory_df: DataFrame with columns ['time', 'theta', 'x', 'action', 'reward']
        save_path: Optional path to save animation (.mp4 or .gif)
        figsize: Figure size (width, height)
        show_angle_plot: Whether to show angle plot alongside animation
        fps: Frames per second for animation

    Returns:
        Animation object (can be displayed in Jupyter with HTML)

    Example:
        >>> from src.evaluation import rollout_rl_timed
        >>> from src.rendering import animate_trajectory
        >>>
        >>> # Run rollout
        >>> traj, _ = rollout_rl_timed(model, vec_env, start_state)
        >>>
        >>> # Animate and save
        >>> anim = animate_trajectory(traj, save_path='runs/demo.mp4')
        >>> # Or just show
        >>> animate_trajectory(traj)  # Shows in window

    Note:
        - Requires ffmpeg for MP4 output
        - Use .gif extension for GIF output (works without ffmpeg)
        - For papers, MP4 is recommended (smaller, better quality)
    """
    if show_angle_plot:
        fig, (ax_anim, ax_angle) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax_anim = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        ax_angle = None

    # --- Animation Panel ---
    ax_anim.set_xlim(-3, 3)
    ax_anim.set_ylim(-0.5, 2.5)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, alpha=0.3)
    ax_anim.set_xlabel('Position (m)', fontsize=12)
    ax_anim.set_ylabel('Height (m)', fontsize=12)
    ax_anim.set_title('Cart-Pendulum Animation', fontsize=14, fontweight='bold')

    # Draw rail
    ax_anim.plot([-2.4, 2.4], [0, 0], 'k-', linewidth=4, label='Rail')

    # Draw rail limits
    ax_anim.axvline(-2.4, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Limits')
    ax_anim.axvline(2.4, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # Cart and pole artists (will be updated)
    cart = Rectangle((-0.25, 0), 0.5, 0.3, fc='steelblue', ec='black', linewidth=2)
    pole_line, = ax_anim.plot([], [], 'r-', linewidth=5)
    pole_bob = Circle((0, 1), 0.08, fc='darkred', ec='black', linewidth=2)
    time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes,
                             verticalalignment='top', fontsize=11,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_anim.add_patch(cart)
    ax_anim.add_patch(pole_bob)
    ax_anim.legend(loc='upper right', fontsize=10)

    # --- Angle Plot Panel ---
    if show_angle_plot:
        ax_angle.set_xlim(0, trajectory_df['time'].max())
        ax_angle.set_ylim(-180, 180)
        ax_angle.set_xlabel('Time (s)', fontsize=12)
        ax_angle.set_ylabel('Angle (degrees)', fontsize=12)
        ax_angle.set_title('Pole Angle Over Time', fontsize=14, fontweight='bold')
        ax_angle.grid(True, alpha=0.3)
        ax_angle.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target')

        angle_line, = ax_angle.plot([], [], 'b-', linewidth=2, label='Angle')
        time_marker = ax_angle.axvline(0, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Current')
        ax_angle.legend(loc='upper right', fontsize=10)
    else:
        angle_line = None
        time_marker = None

    def init():
        """Initialize animation."""
        cart.set_xy((-0.25, 0))
        pole_line.set_data([], [])
        pole_bob.center = (0, 1.15)
        time_text.set_text('')

        if show_angle_plot:
            angle_line.set_data([], [])
            time_marker.set_xdata([0, 0])
            return cart, pole_line, pole_bob, time_text, angle_line, time_marker
        else:
            return cart, pole_line, pole_bob, time_text

    def animate(frame):
        """Update animation for given frame."""
        # Get state at this frame
        x = trajectory_df['x'].iloc[frame]
        theta = trajectory_df['theta'].iloc[frame]
        t = trajectory_df['time'].iloc[frame]

        # Update cart position
        cart.set_xy((x - 0.25, 0))

        # Update pole (theta=0 is up, positive is CCW)
        pole_x = [x, x + np.sin(theta)]
        pole_y = [0.15, 0.15 + np.cos(theta)]
        pole_line.set_data(pole_x, pole_y)

        # Update pole bob (mass at end)
        pole_bob.center = (x + np.sin(theta), 0.15 + np.cos(theta))

        # Update time text
        angle_deg = np.rad2deg(theta)
        time_text.set_text(f't={t:.2f}s\nθ={angle_deg:.1f}°\nx={x:.2f}m')

        # Update angle plot
        if show_angle_plot:
            angle_line.set_data(
                trajectory_df['time'].iloc[:frame+1],
                np.rad2deg(trajectory_df['theta'].iloc[:frame+1])
            )
            time_marker.set_xdata([t, t])
            return cart, pole_line, pole_bob, time_text, angle_line, time_marker
        else:
            return cart, pole_line, pole_bob, time_text

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(trajectory_df),
        interval=1000/fps,  # milliseconds
        blit=True,
        repeat=True
    )

    # Save or show
    if save_path:
        if save_path.endswith('.gif'):
            print(f"Saving GIF to {save_path} (this may take a while)...")
            anim.save(save_path, writer='pillow', fps=fps)
        else:  # Assume MP4
            print(f"Saving MP4 to {save_path}...")
            try:
                anim.save(save_path, writer='ffmpeg', fps=fps, bitrate=1800)
            except Exception as e:
                print(f"Error saving MP4: {e}")
                print("Try installing ffmpeg or save as .gif instead")
                raise
        print(f"Animation saved successfully!")
    else:
        plt.show()

    return anim


def animate_comparison(
    rl_trajectory: pd.DataFrame,
    classical_trajectory: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    fps: int = 50
):
    """
    Create side-by-side comparison animation of RL vs Classical.

    Shows:
    - Top row: RL animation (left), Classical animation (right)
    - Bottom row: Angle comparison plot

    Args:
        rl_trajectory: RL trajectory DataFrame
        classical_trajectory: Classical trajectory DataFrame
        save_path: Optional save path
        figsize: Figure size
        fps: Frames per second

    Returns:
        Animation object

    Example:
        >>> rl_traj, _ = rollout_rl_timed(model, vec_env, start_state)
        >>> classical_traj, _ = rollout_classical_timed(planner, vec_env, start_state)
        >>> animate_comparison(rl_traj, classical_traj, 'comparison.mp4')
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax_rl = fig.add_subplot(gs[0, 0])
    ax_classical = fig.add_subplot(gs[0, 1])
    ax_angle = fig.add_subplot(gs[1, :])

    # Configure animation axes
    for ax, title in [(ax_rl, 'RL Controller'), (ax_classical, 'Classical Controller')]:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position (m)')
        ax.set_title(title, fontweight='bold')
        ax.plot([-2.4, 2.4], [0, 0], 'k-', linewidth=3)
        ax.axvline(-2.4, color='r', linestyle='--', alpha=0.5)
        ax.axvline(2.4, color='r', linestyle='--', alpha=0.5)

    # RL artists
    cart_rl = Rectangle((-0.25, 0), 0.5, 0.3, fc='steelblue', ec='black', lw=2)
    pole_line_rl, = ax_rl.plot([], [], 'r-', linewidth=4)
    pole_bob_rl = Circle((0, 1), 0.08, fc='darkred', ec='black', lw=2)
    ax_rl.add_patch(cart_rl)
    ax_rl.add_patch(pole_bob_rl)

    # Classical artists
    cart_classical = Rectangle((-0.25, 0), 0.5, 0.3, fc='darkorange', ec='black', lw=2)
    pole_line_classical, = ax_classical.plot([], [], 'purple', linewidth=4)
    pole_bob_classical = Circle((0, 1), 0.08, fc='darkviolet', ec='black', lw=2)
    ax_classical.add_patch(cart_classical)
    ax_classical.add_patch(pole_bob_classical)

    # Angle comparison plot
    max_time = max(rl_trajectory['time'].max(), classical_trajectory['time'].max())
    ax_angle.set_xlim(0, max_time)
    ax_angle.set_ylim(-180, 180)
    ax_angle.set_xlabel('Time (s)', fontsize=12)
    ax_angle.set_ylabel('Angle (degrees)', fontsize=12)
    ax_angle.set_title('Angle Comparison', fontsize=14, fontweight='bold')
    ax_angle.grid(True, alpha=0.3)
    ax_angle.axhline(0, color='g', linestyle='--', alpha=0.5)

    angle_line_rl, = ax_angle.plot([], [], 'b-', linewidth=2, label='RL')
    angle_line_classical, = ax_angle.plot([], [], 'orange', linewidth=2, label='Classical')
    time_marker = ax_angle.axvline(0, color='red', alpha=0.7, linewidth=2)
    ax_angle.legend()

    # Determine frame count (use longer trajectory)
    n_frames = max(len(rl_trajectory), len(classical_trajectory))

    def update_cart_pole(ax, cart, pole_line, pole_bob, traj, frame):
        """Helper to update cart-pole visualization."""
        if frame < len(traj):
            x = traj['x'].iloc[frame]
            theta = traj['theta'].iloc[frame]

            cart.set_xy((x - 0.25, 0))
            pole_x = [x, x + np.sin(theta)]
            pole_y = [0.15, 0.15 + np.cos(theta)]
            pole_line.set_data(pole_x, pole_y)
            pole_bob.center = (x + np.sin(theta), 0.15 + np.cos(theta))

    def animate(frame):
        """Update all artists."""
        # Update RL
        update_cart_pole(ax_rl, cart_rl, pole_line_rl, pole_bob_rl, rl_trajectory, frame)

        # Update Classical
        update_cart_pole(ax_classical, cart_classical, pole_line_classical,
                        pole_bob_classical, classical_trajectory, frame)

        # Update angle plots
        if frame < len(rl_trajectory):
            angle_line_rl.set_data(
                rl_trajectory['time'].iloc[:frame+1],
                np.rad2deg(rl_trajectory['theta'].iloc[:frame+1])
            )
            t = rl_trajectory['time'].iloc[frame]
            time_marker.set_xdata([t, t])

        if frame < len(classical_trajectory):
            angle_line_classical.set_data(
                classical_trajectory['time'].iloc[:frame+1],
                np.rad2deg(classical_trajectory['theta'].iloc[:frame+1])
            )

        return (cart_rl, pole_line_rl, pole_bob_rl,
                cart_classical, pole_line_classical, pole_bob_classical,
                angle_line_rl, angle_line_classical, time_marker)

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, blit=True)

    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps, bitrate=1800)
        print(f"Comparison animation saved to {save_path}")
    else:
        plt.show()

    return anim
